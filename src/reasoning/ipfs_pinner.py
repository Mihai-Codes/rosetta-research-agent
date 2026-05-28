"""Pin reasoning traces to IPFS via Pinata.

**Status: stub.** Returns deterministic mock CIDs when ``PINATA_JWT`` is unset
so the rest of the pipeline can run during development. Real Pinata integration
lands in Sprint 1, Day 7.

Why Pinata (vs running our own IPFS node, vs Irys)?
- Free tier covers 1 GB / 10k pins — way more than we'll use in 14 days.
- One JWT, one HTTPS endpoint. Zero infra.
- Irys/Arweave is the post-hackathon backup for permanence guarantees.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import httpx
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

PINATA_API_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"


class IPFSPinError(RuntimeError):
    """Raised when a real Pinata pin attempt fails after retries."""


def _mock_cid(payload: dict[str, Any]) -> str:
    """Deterministic fake CID for offline development."""
    digest = hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).digest()
    # bafy... CID-v1-ish prefix; not a real CID but unambiguous as a mock.
    return "bafymock" + digest.hex()[:48]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _pin_real(payload: dict[str, Any], jwt: str, *, name: str | None = None) -> str:
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
    body = {
        "pinataContent": payload,
        "pinataOptions": {"cidVersion": 1},
        "pinataMetadata": {"name": name or "rosetta-alpha-trace"},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(PINATA_API_URL, headers=headers, json=body)
        resp.raise_for_status()
        cid = resp.json()["IpfsHash"]
        logger.info("Pinned to IPFS: %s (name=%s)", cid, name)
        return cid


async def pin_json(payload: dict[str, Any], *, name: str | None = None) -> str:
    """Pin a JSON-serializable payload. Returns the IPFS CID.

    Falls back to a deterministic mock CID when ``PINATA_JWT`` is not set —
    intentional, so dev loops work offline. The mock prefix makes accidental
    mock-in-prod easy to spot.

    Args:
        payload: JSON-serializable dict to pin.
        name: Optional human-readable label shown in the Pinata dashboard.
    """
    jwt = os.getenv("PINATA_JWT", "").strip()
    if not jwt:
        cid = _mock_cid(payload)
        logger.warning("PINATA_JWT not set — returning mock CID %s", cid)
        return cid

    try:
        return await _pin_real(payload, jwt, name=name)
    except httpx.HTTPError as e:
        raise IPFSPinError(f"Pinata pin failed: {e}") from e
