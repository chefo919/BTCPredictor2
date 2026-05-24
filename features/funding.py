"""
Binance perpetuals funding rate fetcher.
Adds current_funding_rate and avg_funding_24h to the gate snapshot.
"""
import urllib.request
import json


def get_funding_features() -> dict:
    """
    Fetch last 3 funding rate entries from Binance perpetuals.
    Returns {"current_funding_rate": float, "avg_funding_24h": float}.
    Timeout: 3s. On any failure: returns zeros — never blocks the trading loop.
    """
    url = ("https://fapi.binance.com/fapi/v1/fundingRate"
           "?symbol=BTCUSDT&limit=3")
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read().decode())
        rates = [float(entry["fundingRate"]) for entry in data if "fundingRate" in entry]
        if not rates:
            return {"current_funding_rate": 0.0, "avg_funding_24h": 0.0}
        return {
            "current_funding_rate": rates[-1],
            "avg_funding_24h":      sum(rates) / len(rates),
        }
    except Exception:
        return {"current_funding_rate": 0.0, "avg_funding_24h": 0.0}
