import yfinance as yf

def get_nifty_price(date: str):
    """Fetch Nifty 50's closing price for the given date (YYYY-MM-DD)."""
    ticker = "ASIANPAINT.NS"  # Nifty 50 Index symbol on Yahoo Finance
    nifty = yf.Ticker(ticker)

    # Fetch historical market data
    hist = nifty.history(start=date, end=date)
    print(hist)
    if not hist.empty:
        return {"date": date, "closing_price": hist["Close"].iloc[0]}
    else:
        return {"error": "No data available for the given date"}

# Example usage
nifty_price = get_nifty_price("2024-02-01")  # Replace with your desired date
print(nifty_price)
