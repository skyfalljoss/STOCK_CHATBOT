# backend/data_fetcher.py
import yfinance as yf
import requests

# --- Configuration ---
# IMPORTANT: Replace with your own Alpha Vantage API key for news
ALPHA_VANTAGE_API_KEY = "UL8R8I099TCNWANY"

# def get_stock_price(symbol):
#     """Fetches real-time stock price from yfinance."""
#     try:
#         ticker = yf.Ticker(symbol)
#         # Use history() for more robust data fetching
#         data = ticker.history(period="1d", interval="1m")
#         if not data.empty:
#             price = data['Close'].iloc[-1]
#             print("-------------------------")
#             print(f"Fetched price for {symbol}: {price}")
#             return {'symbol': symbol, 'price': price}
#         else:
#             return {'error': f"Could not retrieve recent price data for '{symbol}'."}
#     except Exception as e:
#         return {'error': f"An error occurred while fetching price data for {symbol}."}

def get_company_news(symbol):
    """Fetches company news from Alpha Vantage."""
    if ALPHA_VANTAGE_API_KEY == "HRJ0O8ALOG17MYU8":
        return {'error': "Alpha Vantage API key not configured."}
    try:
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&limit=3&apikey={ALPHA_VANTAGE_API_KEY}'
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        if "feed" in data and data["feed"]:
            articles = [{'title': a['title'], 'url': a['url']} for a in data["feed"][:3]]
            return {'symbol': symbol, 'articles': articles}
        else:
            return {'error': f"Could not find any recent news for '{symbol}'."}
    except Exception:
        return {'error': f"An error occurred while fetching news for {symbol}."}


import requests

def get_stock_price(symbol):
    """
    Fetches the real-time stock price from Alpha Vantage.
    Args:
        symbol (str): The stock ticker symbol.
        api_key (str): Your Alpha Vantage API key.
    Returns:
        dict: Contains 'symbol' and 'price', or 'error' if fetching failed.
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        print(f"Response data for {symbol}: {data}")
        if 'Global Quote' in data and '05. price' in data['Global Quote']:
            price = float(data['Global Quote']['05. price'])
            print("-------------------------")
            print(f"Fetched price for {symbol}: {price}")
            return {'symbol': symbol, 'price': price}
        else:
            return {'error': f"Could not retrieve price data for '{symbol}'."}
    except Exception as e:
        return {'error': f"An error occurred while fetching price data for {symbol}: {str(e)}"}


if __name__ == "__main__":
    # Example usage
    symbol = "AAPL"
    price_data = get_stock_price(symbol)
    print(price_data)
