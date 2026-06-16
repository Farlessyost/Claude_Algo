"""Signed REST client for the Kalshi margin / perpetual-futures API.

Auth follows Kalshi's documented scheme (shared with the trade-api):
    msg = timestamp_ms + HTTP_METHOD + request_path
    signature = base64( RSA-PSS-SHA256(private_key, msg) )
Headers:
    KALSHI-ACCESS-KEY        -> API key id (UUID)
    KALSHI-ACCESS-SIGNATURE  -> base64 signature
    KALSHI-ACCESS-TIMESTAMP  -> milliseconds since epoch

`request_path` is the path part beginning with /trade-api/v2 (no query string).

All amounts in/out are fixed-point strings (USD or contracts). Helpers convert
to/from floats at the edges.
"""
from __future__ import annotations

import base64
import time
from typing import Any, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .config import Credentials, PROD_BASE, DEMO_BASE


class KalshiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"Kalshi API {status}: {body}")
        self.status = status
        self.body = body


def _fp(x: float, decimals: int = 6) -> str:
    return f"{x:.{decimals}f}"


def to_float(s: Any, default: float = 0.0) -> float:
    try:
        if s is None:
            return default
        return float(s)
    except (TypeError, ValueError):
        return default


class KalshiClient:
    def __init__(self, creds: Credentials, environment: str = "production",
                 timeout: float = 15.0):
        self.creds = creds
        self.environment = environment
        self.base = PROD_BASE if environment == "production" else DEMO_BASE
        self._private_key = serialization.load_pem_private_key(
            creds.private_key_pem.encode("utf-8"), password=None
        )
        if not isinstance(self._private_key, rsa.RSAPrivateKey):
            raise ValueError("Provided key is not an RSA private key")
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------ auth
    def _sign(self, method: str, path: str) -> dict:
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{path}".encode("utf-8")
        signature = self._private_key.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.creds.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("ascii"),
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, endpoint: str, *,
                 params: Optional[dict] = None,
                 json_body: Optional[dict] = None) -> dict:
        # The signed path must include the /trade-api/v2 prefix but not query.
        path = f"/trade-api/v2{endpoint}"
        url = f"{self.base}{endpoint}"
        headers = self._sign(method, path)
        resp = self._client.request(method, url, params=params,
                                    json=json_body, headers=headers)
        if resp.status_code >= 400:
            raise KalshiError(resp.status_code, resp.text)
        if not resp.content:
            return {}
        return resp.json()

    # --------------------------------------------------------------- account
    def get_balance(self) -> dict:
        return self._request("GET", "/margin/balance",
                             params={"compute_available_balance": "true"})

    def get_positions(self, ticker: Optional[str] = None) -> dict:
        params = {"ticker": ticker} if ticker else None
        return self._request("GET", "/margin/positions", params=params)

    def get_risk(self) -> dict:
        try:
            return self._request("GET", "/margin/risk")
        except KalshiError:
            return {}

    def get_fills(self, ticker: Optional[str] = None, limit: int = 50) -> dict:
        params: dict = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        try:
            return self._request("GET", "/margin/fills", params=params)
        except KalshiError:
            return {"fills": []}

    # ---------------------------------------------------------------- orders
    def get_orders(self, ticker: Optional[str] = None,
                   status: str = "resting") -> dict:
        params: dict = {"status": status}
        if ticker:
            params["ticker"] = ticker
        try:
            return self._request("GET", "/margin/orders", params=params)
        except KalshiError:
            return {"orders": []}

    def create_order(self, *, ticker: str, side: str, count: float, price: float,
                     client_order_id: str, time_in_force: str = "immediate_or_cancel",
                     reduce_only: bool = False, post_only: bool = False,
                     self_trade_prevention_type: str = "taker_at_cross",
                     subaccount: int = 0) -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,                       # "bid" (long) or "ask" (short)
            "count": _fp(count, 2),
            "price": _fp(price, 4),  # KXBTCPERP tick is 0.0001
            "time_in_force": time_in_force,
            "self_trade_prevention_type": self_trade_prevention_type,
            "subaccount": subaccount,
        }
        if reduce_only:
            body["reduce_only"] = True
        if post_only:
            body["post_only"] = True
        return self._request("POST", "/margin/orders", json_body=body)

    def cancel_order(self, order_id: str) -> dict:
        return self._request("DELETE", f"/margin/orders/{order_id}")

    def amend_order(self, order_id: str, *, price: Optional[float] = None,
                    count: Optional[float] = None) -> dict:
        body: dict = {}
        if price is not None:
            body["price"] = _fp(price, 6)
        if count is not None:
            body["count"] = _fp(count, 2)
        return self._request("POST", f"/margin/orders/{order_id}/amend",
                             json_body=body)

    # ----------------------------------------------------------- market data
    def get_markets(self) -> dict:
        try:
            return self._request("GET", "/margin/markets")
        except KalshiError:
            return {"markets": []}

    def get_market(self, ticker: str) -> dict:
        return self._request("GET", f"/margin/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        return self._request("GET", f"/margin/markets/{ticker}/orderbook",
                             params={"depth": depth})

    def get_candlesticks(self, ticker: str, start_ts: int, end_ts: int,
                         period_interval: int = 1) -> dict:
        return self._request(
            "GET", f"/margin/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts,
                    "period_interval": period_interval},
        )

    def get_fee_tiers(self) -> dict:
        try:
            return self._request("GET", "/margin/fee_tiers")
        except KalshiError:
            return {}

    # --------------------------------------------------------------- funding
    def get_funding_rate_estimate(self, ticker: str) -> dict:
        try:
            return self._request(
                "GET", "/margin/funding/rate-estimate", params={"ticker": ticker})
        except KalshiError:
            return {}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
