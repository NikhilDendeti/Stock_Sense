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


# Need to include the commodities and forex rates as well.
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
EODHD_API_KEY = "67a6e6000d6a32.79767451"

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# --- API Limits ---
ALPHA_VANTAGE_RPM = 5
EODHD_RPM = 10
API_WAIT_TIME = 60  # 1-minute wait between API calls

# --- Initialize LLM Client ---
llm_client = Groq(api_key=GROQ_API_KEY)

# --- Function to Fetch Stock News and Generate Summary ---
@sleep_and_retry
@limits(calls=EODHD_RPM, period=60)
def get_stock_news(symbol: str, api_token: str = EODHD_API_KEY, user_query: str = ""):
    """
    Fetches stock news and generates a focused summary based on the user's query while ensuring the total content does not exceed 27,000 characters.
    """
    time.sleep(API_WAIT_TIME)  # Enforce 60s gap between API calls
    url = f'https://eodhd.com/api/news?s={symbol}.US&offset=0&limit=10&api_token={api_token}&fmt=json'
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ Unable to fetch stock news. Please try again later."
    
    news = response.json()
    content_list = []
    total_chars = 0
    
    for article in news:
        content = article.get("content", "")
        if total_chars + len(content) <= 2500:
            content_list.append(content)
            total_chars += len(content)
    
    if not content_list:
        return "No relevant news articles available at the moment."
    
    summary_prompt = f"Summarize the following stock news focusing on answering the user's query: '{user_query}'\n\n" + "\n".join(content_list)
    response = llm_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": summary_prompt}],
        temperature=0.3,
        top_p=0.9
    )
    summary_response = response.choices[0].message.content.strip()
    
    return f"📢 **Summary of Latest News for {symbol}:**\n\n{summary_response}"

# --- Function to Fetch Commodity Data ---
@sleep_and_retry
@limits(calls=ALPHA_VANTAGE_RPM, period=60)
def get_commodity_data(function: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches commodity data (e.g., WTI, Brent, Natural Gas, Copper) for the last 3 days.
    """
    time.sleep(API_WAIT_TIME)  
    url = f'https://www.alphavantage.co/query?function={function}&interval=daily&apikey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        return f"❌ Unable to fetch data for {function}. Please try again later."

    data = response.json()
    time_series = data.get("Time Series (Daily)", {})
    if not time_series:
        return f"No daily data available for {function}."

    # Get the last 3 days of data
    last_3_days = list(time_series.keys())[:3]
    result = {date: time_series[date] for date in last_3_days}

    return {
        "commodity": function,
        "last_3_days_data": result
    }

# --- Function to Fetch Real GDP Data ---
@sleep_and_retry
@limits(calls=ALPHA_VANTAGE_RPM, period=60)
def get_gdp_data(api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches real GDP data.
    """
    time.sleep(API_WAIT_TIME)  
    url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ Unable to fetch GDP data. Please try again later."
    
    data = response.json()
    return data

# --- Function to Fetch Crypto Exchange Rate ---
@sleep_and_retry
@limits(calls=ALPHA_VANTAGE_RPM, period=60)
def get_crypto_exchange_rate(from_currency: str, to_currency: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches crypto exchange rate.
    """
    time.sleep(API_WAIT_TIME)  
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        return f"❌ Unable to fetch exchange rate for {from_currency} to {to_currency}. Please try again later."
    
    data = response.json()
    return data

# --- Function to Fetch Global Stock Quote ---

@limits(calls=ALPHA_VANTAGE_RPM, period=60)
def get_global_quote(symbol: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches global stock quote for a given symbol.
    """
    time.sleep(API_WAIT_TIME)  
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        return f"❌ Unable to fetch global quote for {symbol}. Please try again later."
    
    data = response.json()
    return data

# --- Register Functions as Tools ---
get_stock_news_tool = FunctionTool.from_defaults(get_stock_news)
get_commodity_data_tool = FunctionTool.from_defaults(get_commodity_data)
get_gdp_data_tool = FunctionTool.from_defaults(get_gdp_data)
get_crypto_exchange_rate_tool = FunctionTool.from_defaults(get_crypto_exchange_rate)
get_global_quote_tool = FunctionTool.from_defaults(get_global_quote)

# --- Initialize LLaMA Model ---
llm = GroqLLM(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# --- Create ReAct Agent ---
agent = ReActAgent.from_tools(
    [
        get_stock_news_tool,
        get_commodity_data_tool,
        get_gdp_data_tool,
        get_crypto_exchange_rate_tool,
        get_global_quote_tool
    ],
    llm=llm,
    verbose=True
)

# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Test Stock News
        print(agent.chat("What is the stock price of reilance industries?"))
        time.sleep(API_WAIT_TIME)
        
        # # Test Commodity Data
        # print(agent.chat("Get the latest data for WTI."))
        # time.sleep(API_WAIT_TIME)
        # print(agent.chat("Get the latest data for Brent."))
        
        # # Test GDP Data
        # print(agent.chat("Get the latest real GDP data."))
        # time.sleep(API_WAIT_TIME)
        
        # # Test Crypto Exchange Rate
        # print(agent.chat("What is the exchange rate between BTC and USD?"))
        # time.sleep(API_WAIT_TIME)
        
        # # Test Global Quote
        # print(agent.chat("What is the current global quote for IBM?"))
    except Exception as e:
        logging.error(f"Error: {str(e)}")
