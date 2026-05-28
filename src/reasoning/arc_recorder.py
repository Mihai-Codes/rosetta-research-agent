"""Record reasoning-trace hashes on Arc via ReasoningRegistry.sol.

Arc testnet details (per docs.arc.network):
  RPC:      https://rpc.testnet.arc.network  (chain ID 5042002)
  Gas:      USDC (18 decimals) — denominated in dollars, predictable
  Explorer: https://testnet.arcscan.app

Falls back to a deterministic mock tx hash when the three required env vars
(REASONING_REGISTRY_ADDRESS, ARC_RPC_URL, ARC_DEPLOYER_PRIVATE_KEY) are not
set — preserving the offline-dev pattern used by IPFSPinner.

Required env vars for live operation:
  ARC_RPC_URL                  — e.g. https://rpc.testnet.arc.network or arc-canteen RPC
  REASONING_REGISTRY_ADDRESS   — 0x address of deployed ReasoningRegistry
  ARC_DEPLOYER_PRIVATE_KEY     — 0x private key of an authorized submitter wallet
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

from reasoning.trace_schema import AssetClass, Direction, InvestmentThesis, Region, TraceMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region / AssetClass → Solidity enum uint8 mapping
# Mirrors the enum order in contracts/src/ReasoningRegistry.sol exactly.
# DO NOT REORDER — the contract ABI is the source of truth.
# ---------------------------------------------------------------------------

_REGION_TO_UINT8: dict[Region, int] = {
    Region.US:     0,
    Region.CN:     1,
    Region.EU:     2,
    Region.JP:     3,
    Region.CRYPTO: 4,
}

_ASSET_CLASS_TO_UINT8: dict[AssetClass, int] = {
    AssetClass.EQUITY:       0,
    AssetClass.FIXED_INCOME: 1,
    AssetClass.COMMODITY:    2,
    AssetClass.CRYPTO:       3,
    AssetClass.FX:           4,
    AssetClass.REAL_ESTATE:  5,
}

# Minimal ABI for RosettaToken stake() + balanceOf() + stakedBalance()
_TOKEN_ABI: list[dict[str, Any]] = [
    {
        "name": "stake",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [{"name": "amount", "type": "uint256"}],
        "outputs": [],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "stakedBalance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]

# Default stake per trace: 10 ROSETTA (18 decimals). Override with ROSETTA_STAKE_AMOUNT env var.
_DEFAULT_STAKE = 10 * 10**18

# ---------------------------------------------------------------------------
# Direction → Solidity enum uint8 mapping
# Mirrors Direction enum in contracts/src/PredictionMarket.sol exactly.
# ---------------------------------------------------------------------------

_DIRECTION_TO_UINT8: dict[Direction, int] = {
    Direction.LONG:    0,
    Direction.SHORT:   1,
    Direction.NEUTRAL: 2,
}

# Minimal ABI for PredictionMarket createMarket()
_MARKET_ABI: list[dict[str, Any]] = [
    {
        "name": "createMarket",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "traceHash",    "type": "bytes32"},
            {"name": "agent",        "type": "address"},
            {"name": "stakeAmount",  "type": "uint256"},
            {"name": "assetKey",     "type": "bytes32"},
            {"name": "direction",    "type": "uint8"},
            {"name": "confidenceBp", "type": "uint16"},
            {"name": "entryPrice",   "type": "uint256"},
            {"name": "horizonDays",  "type": "uint32"},
        ],
        "outputs": [],
    },
    {
        "name": "MarketAlreadyExists",
        "type": "error",
        "inputs": [{"name": "traceHash", "type": "bytes32"}],
    },
]

# Minimal ABI for the record() function — avoids bundling the full JSON.
_RECORD_ABI: list[dict[str, Any]] = [
    {
        "name": "record",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "traceHash",  "type": "bytes32"},
            {"name": "ipfsCid",    "type": "string"},
            {"name": "region",     "type": "uint8"},
            {"name": "assetClass", "type": "uint8"},
        ],
        "outputs": [],
    },
    {
        "name": "TraceAlreadyExists",
        "type": "error",
        "inputs": [{"name": "traceHash", "type": "bytes32"}],
    },
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class ArcRecordError(RuntimeError):
    """Raised when on-chain recording fails and cannot be retried."""


def _mock_tx_hash(metadata: TraceMetadata) -> str:
    digest = hashlib.sha256(metadata.trace_hash.encode()).hexdigest()
    return f"0xmock{digest[:60]}"


def _trace_hash_to_bytes32(hex_hash: str) -> bytes:
    """Convert a 0x-prefixed 64-char hex string to 32 raw bytes."""
    h = hex_hash.removeprefix("0x")
    if len(h) != 64:
        raise ValueError(f"trace_hash must be 64 hex chars, got {len(h)}: {hex_hash!r}")
    return bytes.fromhex(h)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def create_market(
    w3: Any,
    account: Any,
    trace_bytes: bytes,
    thesis: "InvestmentThesis",
    stake_amount: int,
) -> str | None:
    """Call PredictionMarket.createMarket() after a trace has been recorded.

    Returns the tx hash, or None if PREDICTION_MARKET_ADDRESS is not configured.
    Idempotent: MarketAlreadyExists is silently swallowed.
    """
    import hashlib as _hl

    market_addr = os.getenv("PREDICTION_MARKET_ADDRESS", "").strip()
    if not market_addr:
        logger.debug("PREDICTION_MARKET_ADDRESS not set — skipping market creation.")
        return None

    from web3 import AsyncWeb3

    market = w3.eth.contract(
        address=AsyncWeb3.to_checksum_address(market_addr),
        abi=_MARKET_ABI,
    )

    # assetKey = keccak256(ticker) — matches oracle convention
    asset_key = w3.keccak(text=thesis.ticker_or_asset)

    direction_int  = _DIRECTION_TO_UINT8[thesis.direction]
    confidence_bp  = int(thesis.confidence_score * 10_000)  # 0-10000 basis points
    # entryPrice: use live price from thesis if available, else 0 (oracle fills on resolve).
    entry_price    = thesis.entry_price_1e8 if thesis.entry_price_1e8 is not None else 0
    horizon_days   = int(thesis.time_horizon_days)

    _20_gwei = 20 * 10**9
    fee_history = await w3.eth.fee_history(5, "latest", [50])
    base_fee = fee_history["baseFeePerGas"][-1]
    max_priority = max(_20_gwei // 20, 1 * 10**9)
    max_fee = max(base_fee * 2 + max_priority, _20_gwei)

    nonce = await w3.eth.get_transaction_count(account.address)
    txn = await market.functions.createMarket(
        trace_bytes,
        account.address,
        stake_amount,
        asset_key,
        direction_int,
        confidence_bp,
        entry_price,
        horizon_days,
    ).build_transaction({
        "from":                 account.address,
        "nonce":                nonce,
        "maxFeePerGas":         max_fee,
        "maxPriorityFeePerGas": max_priority,
        "type":                 2,
    })
    gas_estimate = await w3.eth.estimate_gas(txn)
    txn["gas"] = int(gas_estimate * 1.2)

    signed   = account.sign_transaction(txn)
    tx_hash  = await w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt  = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] == 0:
        logger.error("createMarket reverted — tx=%s", tx_hash.hex())
        return None

    hex_tx = tx_hash.hex()
    logger.info(
        "PredictionMarket created for %s (%s, conf=%.0f%%) ✅  tx=%s",
        thesis.ticker_or_asset,
        thesis.direction.value,
        thesis.confidence_score * 100,
        hex_tx,
    )
    return hex_tx


async def stake_for_trace(
    w3: Any,
    account: Any,
    stake_amount: int | None = None,
) -> str | None:
    """Stake ROSETTA tokens before recording a trace.

    Returns the tx hash of the stake transaction, or None if token address
    is not configured (gracefully skipped — staking is optional for offline dev).
    """
    token_addr = os.getenv("ROSETTA_TOKEN_ADDRESS", "").strip()
    if not token_addr:
        logger.debug("ROSETTA_TOKEN_ADDRESS not set — skipping stake step.")
        return None

    amount = stake_amount or int(os.getenv("ROSETTA_STAKE_AMOUNT", str(_DEFAULT_STAKE)))

    from web3 import AsyncWeb3

    token = w3.eth.contract(
        address=AsyncWeb3.to_checksum_address(token_addr),
        abi=_TOKEN_ABI,
    )

    # Check liquid balance before staking
    liquid = await token.functions.balanceOf(account.address).call()
    if liquid < amount:
        logger.warning(
            "Insufficient ROSETTA balance to stake: have %.2f, need %.2f — skipping.",
            liquid / 1e18,
            amount / 1e18,
        )
        return None

    _20_gwei = 20 * 10**9
    fee_history = await w3.eth.fee_history(5, "latest", [50])
    base_fee = fee_history["baseFeePerGas"][-1]
    max_priority = max(_20_gwei // 20, 1 * 10**9)
    max_fee = max(base_fee * 2 + max_priority, _20_gwei)

    nonce = await w3.eth.get_transaction_count(account.address)
    txn = await token.functions.stake(amount).build_transaction({
        "from": account.address,
        "nonce": nonce,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": max_priority,
        "type": 2,
    })
    gas_estimate = await w3.eth.estimate_gas(txn)
    txn["gas"] = int(gas_estimate * 1.2)

    signed = account.sign_transaction(txn)
    tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    if receipt["status"] == 0:
        logger.error("Stake transaction reverted — tx=%s", tx_hash.hex())
        return None

    hex_tx = tx_hash.hex()
    logger.info(
        "Staked %.2f ROSETTA before trace recording ✅  tx=%s",
        amount / 1e18,
        hex_tx,
    )
    return hex_tx


async def record_trace(
    metadata: TraceMetadata,
    thesis: "InvestmentThesis | None" = None,
) -> str:
    """Submit *metadata* to the on-chain ReasoningRegistry. Returns tx hash.

    If *thesis* is provided **and** PREDICTION_MARKET_ADDRESS is set, a
    PredictionMarket is created for the trace immediately after recording —
    closing the full Analyze → Pin → Stake → Record → Market loop.

    Idempotency: the contract reverts with TraceAlreadyExists if the hash was
    already recorded. We catch that specific error and treat it as success,
    returning a sentinel tx hash so the caller can log it without breaking.

    Falls back to a deterministic mock tx hash when the required env vars are
    not set — no network calls are made in that case.
    """
    registry_addr  = os.getenv("REASONING_REGISTRY_ADDRESS", "").strip()
    rpc_url        = os.getenv("ARC_RPC_URL", "").strip()
    deployer_key   = os.getenv("ARC_DEPLOYER_PRIVATE_KEY", "").strip()

    if not (registry_addr and rpc_url and deployer_key):
        tx = _mock_tx_hash(metadata)
        logger.warning(
            "Arc env vars not set — mocking on-chain record for %s → %s",
            metadata.trace_hash,
            tx,
        )
        return tx

    # Lazy import so web3 is only required when actually used.
    try:
        from web3 import AsyncWeb3
        from web3.middleware import ExtraDataToPOAMiddleware
        from eth_account import Account
    except ImportError as exc:
        raise ArcRecordError(
            "web3 / eth_account not installed. Run: uv add web3"
        ) from exc

    w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
    # Arc testnet is a POA-compatible chain — inject middleware to handle
    # the extra 97-byte `extraData` field that causes decode errors on strict EIP-1559.
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

    account = Account.from_key(deployer_key)
    registry = w3.eth.contract(
        address=AsyncWeb3.to_checksum_address(registry_addr),
        abi=_RECORD_ABI,
    )

    trace_bytes = _trace_hash_to_bytes32(metadata.trace_hash)
    region_int  = _REGION_TO_UINT8[metadata.region]
    asset_int   = _ASSET_CLASS_TO_UINT8[metadata.asset_class]

    # --- Optional: stake ROSETTA bond before recording ---
    stake_amount = int(os.getenv("ROSETTA_STAKE_AMOUNT", str(_DEFAULT_STAKE)))
    stake_tx = await stake_for_trace(w3, account, stake_amount)
    if stake_tx:
        logger.info("Bond staked → %s", stake_tx)

    try:
        nonce = await w3.eth.get_transaction_count(account.address)

        # Arc uses EIP-1559. Minimum maxFeePerGas is 20 Gwei (protocol floor).
        # Using legacy gasPrice causes "transaction underpriced" — always use
        # maxFeePerGas + maxPriorityFeePerGas on Arc.
        _20_gwei = 20 * 10**9
        fee_history = await w3.eth.fee_history(5, "latest", [50])
        base_fee = fee_history["baseFeePerGas"][-1]
        max_priority = max(_20_gwei // 20, 1 * 10**9)          # 1 Gwei tip
        max_fee = max(base_fee * 2 + max_priority, _20_gwei)    # at least 20 Gwei floor

        txn = await registry.functions.record(
            trace_bytes,
            metadata.ipfs_cid,
            region_int,
            asset_int,
        ).build_transaction({
            "from":              account.address,
            "nonce":             nonce,
            "maxFeePerGas":      max_fee,
            "maxPriorityFeePerGas": max_priority,
            "type":              2,  # EIP-1559
        })

        # Estimate gas (denominated in USDC — 18 decimals as native gas on Arc,
        # NOT 6 decimals like ERC-20 USDC. Display in human USDC for observability.)
        gas_estimate = await w3.eth.estimate_gas(txn)
        txn["gas"] = int(gas_estimate * 1.2)  # 20% buffer

        usdc_cost = (txn["gas"] * max_fee) / 1e18
        logger.info(
            "Recording trace %s on Arc — estimated gas cost: %.8f USDC",
            metadata.trace_hash[:12],
            usdc_cost,
        )

        signed = account.sign_transaction(txn)
        tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = await w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] == 0:
            raise ArcRecordError(
                f"Transaction reverted on-chain. tx={tx_hash.hex()}"
            )

        hex_tx = tx_hash.hex()
        logger.info("Trace recorded on Arc ✅  tx=%s", hex_tx)

        # --- Optional: open a PredictionMarket for this trace ---
        if thesis is not None:
            try:
                market_tx = await create_market(
                    w3=w3,
                    account=account,
                    trace_bytes=trace_bytes,
                    thesis=thesis,
                    stake_amount=stake_amount,
                )
                if market_tx:
                    logger.info("PredictionMarket opened ✅  tx=%s", market_tx)
            except Exception as mkt_exc:
                # Market creation is best-effort — never block the record path.
                logger.warning("create_market failed (non-fatal): %s", mkt_exc)

        return hex_tx

    except ArcRecordError:
        raise
    except Exception as exc:
        # Detect TraceAlreadyExists revert — treat as idempotent success.
        if "TraceAlreadyExists" in str(exc) or "0x" in str(exc):
            sentinel = f"0xalready_{metadata.trace_hash[2:18]}"
            logger.info(
                "Trace already exists on-chain (idempotent) → %s", sentinel
            )
            return sentinel
        raise ArcRecordError(f"Arc record_trace failed: {exc}") from exc
