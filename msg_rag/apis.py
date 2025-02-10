from fastapi import FastAPI, BackgroundTasks, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
import logging
import requests
from dotenv import load_dotenv
from news_processor import NewsPipeline, SimpleNewsQuery, EmbeddingModel, NewsSummarizer
from typing import Dict, Optional
from groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.tools import FunctionTool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
import yfinance as yf
import re
import json

def convert_markdown_to_html(text):
    """Converts markdown bold (**text**) to HTML bold (<b>text</b>)."""
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Stock Market API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    news_pipeline = NewsPipeline(
        news_api_key=os.getenv("NEWS_API_KEY"),
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_url=QDRANT_URL,
        collection_name="Todays_Collection",
        llm_api_key=GROQ_API_KEY
    )
except Exception as e:
    logging.error(f"Failed to initialize NewsPipeline: {str(e)}")
    news_pipeline = None

def run_news_pipeline():
    if not news_pipeline:
        logging.error("NewsPipeline is not initialized")
        return
    
    try:
        logging.info(f"🚀 Running News Pipeline at {datetime.now()}...")
        scraped_articles = news_pipeline.process_news()
        news_pipeline.process_scraped_articles(scraped_articles[:5])
        logging.info("✅ News processing completed")
    except Exception as e:
        logging.error(f"News pipeline execution failed: {str(e)}")

@app.get("/storing_articles")
def fetch_and_process_news(background_tasks: BackgroundTasks):
    """Trigger news processing in the background"""
    if not news_pipeline:
        raise HTTPException(status_code=500, detail="News pipeline is not available")
    
    background_tasks.add_task(run_news_pipeline)
    return {"message": "News processing started in the background."}

class ReportQuery(BaseModel):
    user_query: str

@app.get("/daily-report")
async def generate_daily_report():
    if not news_pipeline:
        raise HTTPException(status_code=500, detail="News pipeline is not available")
    
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
        simple_news_query = SimpleNewsQuery(qdrant_client, "Todays_Collection")
        embedding = EmbeddingModel()
        vector = embedding.get_embedding("Indian stock market news today: Sensex, Nifty, top gainers, top losers, Q3 results, earnings, stock movements, macroeconomic updates, sector performance")
        current_datetime = datetime.now()
        news_results = simple_news_query.query_news(vector, current_datetime.strftime("%Y-%m-%d"), limit=8)

        llm_client = Groq(api_key=GROQ_API_KEY)
        news_summarizer = NewsSummarizer(llm_client)
        print(f"📢 Queried News Articles:\n{news_results}")

        summarized_news = news_summarizer.generate_summary(news_results)

        print(f"📢 Stock Market Summary:\n{summarized_news}")
        formatted_news = convert_markdown_to_html(summarized_news)

        return JSONResponse(content=formatted_news)
    except Exception as e:
        logging.error(f"Failed to generate daily report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate daily report")

if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in environment variables")
llm_client = Groq(api_key=GROQ_API_KEY)



# Consrtaints for News Summary
MAX_CHARS = 2500  
MAX_DAYS = 5 

def extract_symbol_from_query(user_query: str, query_type: str):
    """
    Uses Llama3 to extract the correct symbol (stock ticker, commodity name, or crypto symbol) from user input.
    Only applies if the function requires a symbol for an API request.
    """
    prompt = (
        f"You are an intelligent finance assistant. The user is asking about {query_type}."
        f"Extract the correct symbol or name from their query.\n\n"
        f"User Query: '{user_query}'\n\n"
        f"Respond with only the symbol or name, nothing else."
    )

    try:
        response = llm_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.3,
            top_p=0.9
        )
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Symbol extraction error: {str(e)}")
        return None


def get_stock_news(user_query: str, api_token: str = EODHD_API_KEY):
    """
    Fetches stock news only if a valid stock ticker is extracted.
    """
    symbol = extract_symbol_from_query(user_query, "a stock ticker")
    if not symbol:
        return "⚠️ Unable to determine the stock symbol from your query. Please try again."

    url = f'https://eodhd.com/api/news?s={symbol}.US&offset=0&limit=10&api_token={api_token}&fmt=json'
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        news = response.json()
    except requests.RequestException as e:
        logging.error(f"Stock news API error: {str(e)}")
        return "❌ Unable to fetch stock news. Please try again later."
    
    content_list, total_chars = [], 0
    for article in news:
        content = article.get("content", "")
        if total_chars + len(content) <= MAX_CHARS:
            content_list.append(content)
            total_chars += len(content)
    
    if not content_list:
        return f"ℹ️ No relevant news articles found for {symbol} at the moment."

    summary_prompt = (
        f"You are a stock market expert chatbot assisting a user with stock updates.\n"
        f"The user asked: '{user_query}'.\n"
        f"Summarize the following stock news while focusing on answering the user's question:\n\n"
        + "\n".join(content_list)
    )
    
    try:
        response = llm_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": summary_prompt}],
            temperature=0.3,
            top_p=0.9
        )
        summary_response = response.choices[0].message.content.strip()
        text = f"📢 **Summary of Latest News for {symbol}:**\n\n{summary_response}"
        return {
            "summary": convert_markdown_to_html(text),
        }
        
    except Exception as e:
        logging.error(f"LLM summarization error: {str(e)}")
        return "⚠️ Unable to generate a news summary at this time. Please check back later."



def convert_stock_data_for_chart(raw_data, symbol):
    """
    Converts raw stock price data into a format suitable for charting.
    - "name" will be the formatted date (Month Day).
    - "value" will be the closing price.
    
    :param raw_data: Dictionary containing stock price data.
    :return: List of dictionaries formatted for chart usage.
    """
    # Extract closing prices
    closing_prices = raw_data.get(('Close', symbol), {})

    if not closing_prices:
        return []

    # Convert to required format
    formatted_data = [
        {
            "name": date.strftime("%b %d"),  # Format as "Feb 04"
            "value": round(price, 2)  # Round price to 2 decimal places
        }
        for date, price in closing_prices.items()
    ]

    return {
        "stock_price": formatted_data,
    }

def get_stock_price(user_query: str):
    """
    Fetches stock price data for the last 5 days using Yahoo Finance.
    """
    symbol = extract_symbol_from_query(user_query, "a stock ticker")
    if not symbol:
        return "⚠️ Unable to determine the stock symbol from your query. Please try again."
    
    try:
        df = yf.download(symbol, period="5d")
        if df.empty:
            return f"❌ No stock data found for {symbol}."
        print(df.tail(5).to_dict())
        data = convert_stock_data_for_chart(df.tail(5).to_dict(), symbol)
        print(data)
        return data
    except Exception as e:
        logging.error(f"Yahoo Finance API error: {str(e)}")
        return f"❌ Unable to fetch stock price data for {symbol}. Please try again later."
    

VALID_COMMODITIES = {
    "COPPER", "NATURAL_GAS", "BRENT", "WTI", "ALUMINUM",
    "WHEAT", "CORN", "COTTON", "SUGAR", "COFFEE", "ALL_COMMODITIES"
}


def extract_commodity_from_query(user_query: str) -> str:
    """
    Uses Llama3 to extract the correct commodity name from the user query.
    Ensures the extracted name matches one of the valid commodities.
    If no valid commodity is found, defaults to 'ALL_COMMODITIES'.
    """
    prompt = (
        f"You are an expert in financial markets. The user is asking about commodities.\n"
        f"Extract the commodity name from the query. Ensure it matches one of these valid names:\n"
        f"{', '.join(VALID_COMMODITIES)}\n\n"
        f"User Query: '{user_query}'\n\n"
        f"Respond with only the exact commodity name from the list above. If none match, respond with 'ALL_COMMODITIES'."
    )

    try:
        response = llm_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            top_p=1
        )
        commodity = response.choices[0].message.content.strip().upper()

        # Ensure extracted commodity is valid
        return commodity if commodity in VALID_COMMODITIES else "ALL_COMMODITIES"
    
    except Exception as e:
        logging.error(f"Commodity extraction error: {str(e)}")
        return "ALL_COMMODITIES"  # Default fallback


def get_commodity_data(user_query: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Extracts the commodity name from the user query and fetches its price data.
    Returns data for the last 3 available dates.
    """
    # Step 1: Extract commodity name
    commodity = extract_commodity_from_query(user_query)

    # Step 2: Construct API URL
    url = f'https://www.alphavantage.co/query?function={commodity}&interval=daily&apikey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logging.error(f"Commodity data API error: {str(e)}")
        return f"❌ Unable to fetch data for {commodity}. Please try again later."

    # Step 3: Extract only the last 3 dates
    time_series = data.get("data", [])  # Adjust based on Alpha Vantage JSON response format
    last_3_dates = time_series[:5] if len(time_series) >= 5 else time_series
    print(last_3_dates)

    return {
        "commodity": commodity,
        "last_5_dates_data": last_3_dates
    }



def get_crypto_exchange_rate(user_query: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Extracts the correct cryptocurrency pair from the user query and fetches exchange rate.
    If no specific pair is given, fetches general crypto market data.
    """
    pair = extract_symbol_from_query(user_query, "a cryptocurrency pair (e.g., BTC/USD, ETH/USDT)")
    
    if pair and "/" in pair:
        from_currency, to_currency = pair.split("/")
        url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
    else:
        url = f'https://www.alphavantage.co/query?function=CRYPTO_MARKET&apikey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logging.error(f"Crypto exchange API error: {str(e)}")
        return "❌ Unable to fetch crypto data. Please try again later."

    return {
        "user_query": user_query,
        "crypto_data": data
    }

def get_gdp_data(api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches real GDP data.
    """
    url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={api_key}'
    response = requests.get(url)
    if response.status_code != 200:
        return "❌ Unable to fetch GDP data. Please try again later."
    
    data = response.json()
    return data

def get_global_quote(user_query: str, api_key: str = ALPHA_VANTAGE_API_KEY):
    """
    Fetches a stock quote only if a valid stock ticker is extracted.
    """
    symbol = extract_symbol_from_query(user_query, "a stock ticker")
    if not symbol:
        return "⚠️ Unable to determine the stock symbol from your query. Please try again."

    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logging.error(f"Global stock quote API error: {str(e)}")
        return f"❌ Unable to fetch global quote for {symbol}. Please try again later."

    return {
        "stock_quote": data
    }
# Initialize ReAct Agent
try:
    agent = ReActAgent.from_tools(
        [
            FunctionTool.from_defaults(get_stock_news),
            FunctionTool.from_defaults(get_commodity_data),
            FunctionTool.from_defaults(get_crypto_exchange_rate),
            FunctionTool.from_defaults(get_gdp_data),
            FunctionTool.from_defaults(get_stock_price),
        ],
        llm=GroqLLM(model="llama3-70b-8192", api_key=GROQ_API_KEY),
        verbose=True
    )
except Exception as e:
    logging.error(f"Failed to initialize ReAct Agent: {str(e)}")
    agent = None



FUNCTIONS = {
    "stock_news": get_stock_news,
    "commodity_data": get_commodity_data,
    "crypto_exchange_rate": get_crypto_exchange_rate,
    "gdp_data": get_gdp_data,
    "stock_price": get_stock_price,
}


def classify_query(user_query: str) -> Optional[str]:
    """
    Uses Llama3 to classify the user's query and map it to a function.
    Returns the function key or None if the query is irrelevant.
    """
    prompt = (
        f"You are a finance chatbot. Based on the user's query, classify it into one of these categories:\n"
        f"- 'stock_news' (if the query is about stock-related news)\n"
        f"- 'commodity_data' (if the query is about commodities like gold, oil, etc.)\n"
        f"- 'crypto_exchange_rate' (if the query is about crypto exchange rates like BTC/USD)\n"
        f"- 'global_stock_quote' (if the query is about stock prices)\n"
        f"- 'gdp_data' (if the query is about GDP data)\n\n"
        f"- 'stock_price' (if the query is asking about stock prices)\n"
        f"Otherwise, respond with 'none'.\n\n"
        f"User Query: '{user_query}'\n\n"
        f"Respond with only one word: the category name or 'none'."
    )

    try:
        response = llm_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
            top_p=1
        )
        category = response.choices[0].message.content.strip().lower()
        return category if category in FUNCTIONS else None
    
    except Exception as e:
        logging.error(f"Query classification error: {str(e)}")
        return None

# API Endpoint for Chatbot Queries
class UserQuery(BaseModel):
    query: str


@app.post("/query")
async def process_query(user_query: UserQuery) -> Dict[str, str]:
    """
    Handles user queries dynamically and routes them to the correct function.
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Chat service unavailable")

    # Step 1: Classify the query
    function_key = classify_query(user_query.query)
    if not function_key:
        return {"response": "❌ I'm sorry, but I can't help with that. Try asking about stocks, commodities, or crypto."}

    # Step 2: Call the relevant function dynamically
    function_to_call = FUNCTIONS[function_key]
    try:
        response = function_to_call(user_query.query)
        return {"response": str(response)}
    except Exception as e:
        logging.error(f"Function execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process query")


# Scheduler Setup
scheduler = BackgroundScheduler()
try:
    scheduler.add_job(run_news_pipeline, "cron", hour=13, minute=2)
    scheduler.start()
except Exception as e:
    logging.error(f"Failed to start scheduler: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    if scheduler.running:
        scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
