"""Multi-provider IPFS pinning with quorum logic and CID equality assertion.

Pins reasoning traces to multiple IPFS providers in parallel (asyncio.gather),
asserts content-addressing invariant (same payload → same CID across all providers),
and returns structured PinReceipt objects for off-chain audit persistence.

Providers:
  - Pinata (existing): commercial IPFS pinning
  - Storacha (new): Filecoin-backed hot storage via web3.storage successor

Usage:
    multi = MultiPinner(
        pinners=[PinataPinner(jwt), StorachaPinner(sidecar_url)],
        require=1,  # TODO: bump to 2 for production once Storacha is stable
    )
    cid, receipts = await multi.pin(trace_payload, name="us-AAPL-thesis")
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PINATA_FILE_API_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"


def _canonicalize(payload: dict[str, Any]) -> bytes:
    """Produce deterministic canonical JSON bytes for a payload.

    Uses orjson with OPT_SORT_KEYS for consistent key ordering.
    Both Pinata and Storacha receive these exact bytes so that the
    IPFS DAG chunking produces identical CIDs (content-addressing invariant).
    """
    return orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)


# ---------------------------------------------------------------------------
# PinReceipt — structured result from any pinning provider
# ---------------------------------------------------------------------------


@dataclass
class PinReceipt:
    """One pinning attempt result, stored in Postgres pin_receipts table."""

    provider: str  # "pinata" | "storacha"
    cid: str
    pinned_at: int  # unix timestamp
    provider_ref: str | None  # Pinata pin ID, Storacha shard CID, etc.
    status: str  # "ok" | "failed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pinner Protocol — interface for any IPFS pinning provider
# ---------------------------------------------------------------------------


class Pinner(Protocol):
    """Protocol for IPFS pinning providers."""

    async def pin_json(self, payload: dict[str, Any], *, name: str | None = None) -> PinReceipt:
        ...


# ---------------------------------------------------------------------------
# PinataPinner — wraps existing Pinata integration
# ---------------------------------------------------------------------------


class PinataPinner:
    """Pins canonical JSON bytes to IPFS via Pinata's pinFileToIPFS endpoint.

    Uses pinFileToIPFS (not pinJSONToIPFS) so we control the exact bytes that
    get DAG-chunked. This ensures the resulting CID matches other providers
    that receive the same canonical bytes.
    """

    def __init__(self, jwt: str | None = None):
        self.jwt = jwt or os.getenv("PINATA_JWT", "").strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _pin(self, canonical_bytes: bytes, name: str | None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.jwt}"}
        # pinFileToIPFS uses multipart form data
        files = {
            "file": ("trace.json", canonical_bytes, "application/json"),
        }
        data = {
            "pinataOptions": _json.dumps({"cidVersion": 1, "wrapWithDirectory": False}),
            "pinataMetadata": _json.dumps({"name": name or "rosetta-alpha-trace"}),
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                PINATA_FILE_API_URL,
                headers=headers,
                files=files,
                data=data,
            )
            resp.raise_for_status()
            return resp.json()

    async def pin_json(self, payload: dict[str, Any], *, name: str | None = None) -> PinReceipt:
        if not self.jwt:
            logger.warning("PINATA_JWT not set — PinataPinner returning failed receipt")
            return PinReceipt(
                provider="pinata",
                cid="",
                pinned_at=int(time.time()),
                provider_ref=None,
                status="failed",
                error="PINATA_JWT not configured",
            )
        try:
            canonical_bytes = _canonicalize(payload)
            data = await self._pin(canonical_bytes, name)
            return PinReceipt(
                provider="pinata",
                cid=data["IpfsHash"],
                pinned_at=int(time.time()),
                provider_ref=str(data.get("id", "")),
                status="ok",
            )
        except Exception as exc:
            logger.error("PinataPinner failed: %s", exc)
            return PinReceipt(
                provider="pinata",
                cid="",
                pinned_at=int(time.time()),
                provider_ref=None,
                status="failed",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# StorachaPinner — HTTP client to the Storacha Node sidecar
# ---------------------------------------------------------------------------


class StorachaPinner:
    """Pins canonical JSON bytes via the Storacha sidecar (Node.js Express server).

    Sends pre-canonicalized bytes directly so the sidecar uploads the exact
    same content that Pinata receives — ensuring CID equality.
    """

    def __init__(self, sidecar_url: str | None = None):
        self.sidecar_url = (
            sidecar_url or os.getenv("STORACHA_SIDECAR_URL", "http://localhost:3030")
        ).rstrip("/")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _upload(self, canonical_bytes: bytes, name: str | None) -> dict[str, Any]:
        headers = {"Content-Type": "application/octet-stream"}
        if name:
            headers["X-Pin-Name"] = name
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.sidecar_url}/upload",
                content=canonical_bytes,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def pin_json(self, payload: dict[str, Any], *, name: str | None = None) -> PinReceipt:
        try:
            canonical_bytes = _canonicalize(payload)
            data = await self._upload(canonical_bytes, name)
            return PinReceipt(
                provider="storacha",
                cid=data["cid"],
                pinned_at=data.get("pinned_at", int(time.time())),
                provider_ref=data.get("shard"),
                status="ok",
            )
        except Exception as exc:
            logger.error("StorachaPinner failed: %s", exc)
            return PinReceipt(
                provider="storacha",
                cid="",
                pinned_at=int(time.time()),
                provider_ref=None,
                status="failed",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# MultiPinner — parallel pinning with quorum and CID equality assertion
# ---------------------------------------------------------------------------


class CIDMismatchError(RuntimeError):
    """Raised when providers return different CIDs for the same payload."""


class PinQuorumError(RuntimeError):
    """Raised when fewer than `require` providers succeed."""


class MultiPinner:
    """Pins to multiple IPFS providers in parallel with quorum enforcement.

    Args:
        pinners: List of Pinner implementations to pin to in parallel.
        require: Minimum number of successful pins to proceed (default: 1).
                 # TODO: bump to 2 for production once Storacha is stable in the stack
    """

    def __init__(self, pinners: list[Pinner], require: int = 1):
        self.pinners = pinners
        self.require = require

    async def pin(
        self, payload: dict[str, Any], *, name: str | None = None
    ) -> tuple[str, list[PinReceipt]]:
        """Pin payload to all providers in parallel.

        Returns:
            Tuple of (cid, list_of_all_receipts).

        Raises:
            PinQuorumError: If fewer than `require` providers succeed.
            CIDMismatchError: If successful providers disagree on the CID.
        """
        # Fire all pinners in parallel via asyncio.gather
        receipts: list[PinReceipt] = await asyncio.gather(
            *(pinner.pin_json(payload, name=name) for pinner in self.pinners),
        )

        successful = [r for r in receipts if r.status == "ok"]

        if len(successful) < self.require:
            raise PinQuorumError(
                f"Pin quorum failed: need {self.require}, got {len(successful)}. "
                f"Receipts: {[r.to_dict() for r in receipts]}"
            )

        # Content-addressing invariant: all successful pins MUST return the same CID.
        # If they diverge, something is wrong (different chunking, codec, or payload mutation).
        cids = {r.cid for r in successful}
        if len(cids) > 1:
            raise CIDMismatchError(
                f"CID mismatch across providers! Got {cids}. "
                f"This indicates a content-addressing invariant violation. "
                f"Receipts: {[r.to_dict() for r in successful]}"
            )

        cid = cids.pop()
        logger.info(
            "MultiPinner: pinned to %d/%d providers → CID %s",
            len(successful),
            len(receipts),
            cid,
        )
        return cid, receipts


# ---------------------------------------------------------------------------
# Factory — build a configured MultiPinner from environment
# ---------------------------------------------------------------------------


def build_multi_pinner() -> MultiPinner:
    """Construct a MultiPinner from environment variables.

    Env vars:
        PINATA_JWT: Pinata API token
        STORACHA_SIDECAR_URL: URL of the Storacha Node sidecar (default: http://localhost:3030)
    """
    pinners: list[Pinner] = [
        PinataPinner(),
        StorachaPinner(),
    ]
    return MultiPinner(
        pinners=pinners,
        require=1,  # TODO: bump to 2 for production once Storacha is stable
    )
