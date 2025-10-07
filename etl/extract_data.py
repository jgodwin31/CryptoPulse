import requests
import time
import logging
from typing import List, Dict

API_URL = "https://api.coingecko.com/api/v3/coins/markets"

logger = logging.getLogger(__name__)

def fetch_top_coins(vs_currency: str = "usd", per_page: int = 10, page: int = 1, retry: int = 3, timeout: int = 10) -> List[Dict]:
    """
    Fetch top coins by market cap from CoinGecko API.
    Returns list of coin market dicts (one per coin).
    """
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": page,
        "sparkline": "false",
        "price_change_percentage": "24h"
    }

    for attempt in range(1, retry + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            logger.debug("Fetched %d coins from CoinGecko", len(data))
            return data
        except requests.RequestException as e:
            logger.warning("Attempt %d: API request failed: %s", attempt, e)
            if attempt < retry:
                time.sleep(2 ** attempt)
            else:
                raise

if __name__ == "__main__":
    # quick manual test
    import pprint
    logging.basicConfig(level=logging.INFO)
    coins = fetch_top_coins()
    pprint.pprint(coins)
