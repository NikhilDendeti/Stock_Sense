# import requests
# from groq import Groq
# from llama_index.core.tools import FunctionTool
# import os

# # Set API Key
# GROQ_API_KEY = "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"

# def correct_stock_symbol(user_input: str):
#     client = Groq(api_key=GROQ_API_KEY)
#     prompt = f"""
#     You are a financial assistant with expertise in stock symbols.
#     Given a potentially incorrect stock symbol entered by a user, return ONLY the corrected stock symbol.
#     Do not add extra text. If the input is invalid, return 'INVALID'.
    
#     Example:
#     User Input: 'AAPL'
#     Corrected Symbol: 'AAPL'
    
#     User Input: 'Googel'
#     Corrected Symbol: 'GOOGL'
    
#     User Input: 'Microsft'
#     Corrected Symbol: 'MSFT'
    
#     User Input: 'Amzon'
#     Corrected Symbol: 'AMZN'
    
#     User Input: 'Relincent stocks'
#     Corrected Symbol: 'RELIANCE'
    
#     User Input: '{user_input}'
#     Corrected Symbol: """
    
#     completion = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[{"role": "system", "content": prompt}],
#         temperature=0.3,
#         max_completion_tokens=10,
#         top_p=1,
#         stream=False,
#         stop=["\n"]
#     )

#     corrected_symbol = completion.choices[0].message.content.strip()
#     print(f"LLM Returned: {corrected_symbol}")  # Debugging line

#     if corrected_symbol == "INVALID" or not corrected_symbol.isalnum():
#         print(f"Error: Invalid stock symbol received: {corrected_symbol}")
#         return None

#     return corrected_symbol

# correct_stock_symbol_tool = FunctionTool.from_defaults(correct_stock_symbol)

# def validate_api_response(response):
#     if response.status_code == 401:
#         return {"error": "Unauthenticated: Invalid API Key"}
#     if response.status_code == 403:
#         return {"error": "Forbidden: API Access Denied"}
#     if response.status_code != 200:
#         return {"error": f"API Error: {response.status_code}"}
    
#     try:
#         return response.json()
#     except requests.exceptions.JSONDecodeError:
#         return {"error": "Invalid response from API"}

# def get_stock_news(symbol: str, api_token: str):
#     corrected_symbol = correct_stock_symbol_tool(symbol)
#     if not corrected_symbol:
#         return {"error": "Invalid stock symbol"}
    
#     print(f"Original Symbol: {symbol}, Corrected Symbol: {corrected_symbol}")
#     url = f'https://eodhd.com/api/news?s={corrected_symbol}.US&offset=0&limit=10&api_token={api_token}&fmt=json'
#     response = requests.get(url)
#     return validate_api_response(response)

# get_stock_news_tool = FunctionTool.from_defaults(get_stock_news)

# def get_daily_stock_prices(symbol: str, api_key: str):
#     corrected_symbol = correct_stock_symbol_tool(symbol)
#     if not corrected_symbol:
#         return {"error": "Invalid stock symbol"}
    
#     url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={corrected_symbol}&apikey={api_key}'
#     response = requests.get(url)
#     return validate_api_response(response)

# get_daily_stock_prices_tool = FunctionTool.from_defaults(get_daily_stock_prices)

# def get_global_quote(symbol: str, api_key: str):
#     corrected_symbol = correct_stock_symbol_tool(symbol)
#     if not corrected_symbol:
#         return {"error": "Invalid stock symbol"}
    
#     url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={corrected_symbol}&apikey={api_key}'
#     response = requests.get(url)
#     return validate_api_response(response)

# get_global_quote_tool = FunctionTool.from_defaults(get_global_quote)

# def get_crypto_exchange_rate(from_currency: str, to_currency: str, api_key: str):
#     url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
#     response = requests.get(url)
#     return validate_api_response(response)

# get_crypto_exchange_rate_tool = FunctionTool.from_defaults(get_crypto_exchange_rate)

# def get_gdp_data(api_key: str):
#     url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={api_key}'
#     response = requests.get(url)
#     return validate_api_response(response)

# get_gdp_data_tool = FunctionTool.from_defaults(get_gdp_data)

# # Example usage (Replace 'your_api_key' with actual API key):
# print(get_stock_news_tool("AAPL", "67a6e6000d6a32.79767451"))
# print(get_daily_stock_prices_tool("IBM", "PBCPW7HGMTS82RJ6"))
# print(get_global_quote_tool("IBM", "PBCPW7HGMTS82RJ6"))
# print(get_crypto_exchange_rate_tool("BTC", "EUR", "PBCPW7HGMTS82RJ6"))
# print(get_gdp_data_tool("PBCPW7HGMTS82RJ6"))


import requests
import time
import logging
from groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.tools import FunctionTool
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_fixed
from typing import Any, Dict

# Set API Keys
GROQ_API_KEY = "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"
ALPHA_VANTAGE_API_KEY = "PBCPW7HGMTS82RJ6"
EODHD_API_KEY = " 67a6e6000d6a32.79767451"

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# --- API LIMITS ---
## Groq API Limits
GROQ_TPM_LIMIT = 6000  # Max 6000 tokens per minute
GROQ_REQUEST_LIMIT = 30  # Max 30 requests per minute
GROQ_WAIT_TIME = 60  # If limit is reached, wait for 1 minute before retrying

## Alpha Vantage Limits (Free Tier)
ALPHA_VANTAGE_RPM = 5  # Max 5 requests per minute
ALPHA_VANTAGE_WAIT_TIME = 60 / ALPHA_VANTAGE_RPM  # Auto wait to respect limit

## EODHD Limits (Unknown official limit, assuming 10 requests per minute)
EODHD_RPM = 10
EODHD_WAIT_TIME = 60 / EODHD_RPM


# --- Initialize LLM Client ---
llm_client = Groq(api_key=GROQ_API_KEY)


# --- Function to Handle Stock Symbol Correction ---
@sleep_and_retry
@limits(calls=GROQ_REQUEST_LIMIT, period=60)  # Limit: 30 requests per minute
@retry(stop=stop_after_attempt(5), wait=wait_fixed(GROQ_WAIT_TIME))  # Retry if failed, wait for reset
def correct_stock_symbol(user_input: str):
    """
    Uses Groq LLM to correct a stock symbol.
    If the request exceeds token limits, it will wait for 1 minute and retry.
    """
    prompt = f"""
    You are a financial assistant with expertise in stock symbols.
    Given a potentially incorrect stock symbol entered by a user, return ONLY the corrected stock symbol.
    Do not add extra text. If the input is invalid, return 'INVALID'.

    User Input: '{user_input}'
    Corrected Symbol: """

    logging.info("Sending request to Groq for stock symbol correction...")
    
    response = llm_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.3,
        top_p=0.9
    )

    corrected_symbol = response.choices[0].message.content.strip()
    return corrected_symbol if corrected_symbol.isalnum() else "INVALID"


# --- Function to Fetch Stock News with Rate Limiting ---
@sleep_and_retry
@limits(calls=EODHD_RPM, period=60)  # Limit: 10 requests per minute
def get_stock_news(symbol: str, api_token: str = EODHD_API_KEY):
    """
    Fetches stock news with a strict rate limit of 10 requests per minute.
    If exceeded, it waits for 60 seconds before retrying.
    """
    logging.info(f"Fetching news for {symbol}...")
    url = f'https://eodhd.com/api/news?s={symbol}.US&offset=0&limit=10&api_token={api_token}&fmt=json'
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch stock news"}


# --- Function to Fetch Daily Stock Prices with Rate Limiting ---
@sleep_and_retry
@limits(calls=ALPHA_VANTAGE_RPM, period=60)  # Limit: 5 requests per minute
def get_daily_stock_prices(symbol: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches daily stock prices with a strict rate limit of 5 requests per minute.
    If exceeded, it waits for 60 seconds before retrying.
    """
    logging.info(f"Fetching stock prices for {symbol}...")
    time.sleep(ALPHA_VANTAGE_WAIT_TIME)  # Prevents exceeding API rate
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    print(response)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch stock prices"}


# --- Function to Fetch Crypto Exchange Rate with Rate Limiting ---
@sleep_and_retry
@limits(calls=ALPHA_VANTAGE_RPM, period=60)  # Limit: 5 requests per minute
def get_crypto_exchange_rate(from_currency: str, to_currency: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches crypto exchange rate with a strict rate limit of 5 requests per minute.
    If exceeded, it waits for 60 seconds before retrying.
    """
    logging.info(f"Fetching exchange rate from {from_currency} to {to_currency}...")
    time.sleep(ALPHA_VANTAGE_WAIT_TIME)  # Prevents exceeding API rate
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
    response = requests.get(url)
    return response.json() if response.status_code == 200 else {"error": "Failed to fetch exchange rate"}


# --- Register Functions as Tools ---
correct_stock_symbol_tool = FunctionTool.from_defaults(correct_stock_symbol)
get_stock_news_tool = FunctionTool.from_defaults(get_stock_news)
get_daily_stock_prices_tool = FunctionTool.from_defaults(get_daily_stock_prices)
get_crypto_exchange_rate_tool = FunctionTool.from_defaults(get_crypto_exchange_rate)


# --- Initialize LLaMA Model ---
llm = GroqLLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# --- Create ReAct Agent ---
agent = ReActAgent.from_tools(
    [get_stock_news_tool, get_daily_stock_prices_tool, get_crypto_exchange_rate_tool],
    llm=llm,
    verbose=False
)

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # print(agent.chat("What is the latest news about Infosys stock?"))
        print(agent.chat("Get the current price for IBM."))
        print(agent.chat("What's the exchange rate between BTC and EUR?"))
    except Exception as e:
        logging.error(f"Error: {str(e)}")
