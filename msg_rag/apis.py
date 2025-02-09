from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from news_processor import NewsPipeline
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.tools import FunctionTool

app = FastAPI()

# Initialize NewsPipeline
news_pipeline = NewsPipeline(
    news_api_key="7549ec8c3f790b338e0e57e8f5014c1ac1782714",
    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2MTc3MDAyfQ.v-Kra-ZRUdTAe0OBCmCjEvZi8rW_HKY0Cw2Vp-AM5g4",
    qdrant_url="https://3f0e0b2d-447a-48bd-8c2f-3a227ff85295.eu-west-1-0.aws.cloud.qdrant.io:6333/",
    collection_name="Final_News_Articles_Collection",
    llm_api_key="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"
)

@app.get("/storing_articles")
def fetch_and_process_news(background_tasks: BackgroundTasks):
    """API endpoint to trigger the news pipeline in the background."""
    print("API triggered to start news processing in the background...")
    background_tasks.add_task(run_news_pipeline)
    return {"message": "News processing started in the background."}

def run_news_pipeline():
    print(f"🚀 Running News Pipeline at {datetime.now()}...")
    scraped_articles = news_pipeline.process_news()
    total_articles = 5
    batch_size = 2
    wait_time = 180 

    points = []

    for i in range(0, total_articles, batch_size):
        batch = scraped_articles[i:i + batch_size]
        print(f"🚀 Processing batch {i // batch_size + 1}: {len(batch)} articles")
        
        temp_points = news_pipeline.process_scraped_articles(batch)
        points.extend(temp_points)

        if i + batch_size < total_articles:
            print(f"⏳ Waiting for {wait_time // 60} minutes before the next batch...")
            time.sleep(wait_time)
    
    news_pipeline.qdrant.upsert_points(points)
    print("✅ News processing completed and stored in Qdrant.")

@app.get("/get_news_summary")
def get_news_summary():
    qdrant_client = QdrantClient("https://3f0e0b2d-447a-48bd-8c2f-3a227ff85295.eu-west-1-0.aws.cloud.qdrant.io:6333/", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2MTc3MDAyfQ.v-Kra-ZRUdTAe0OBCmCjEvZi8rW_HKY0Cw2Vp-AM5g4")
    simple_news_query = SimpleNewsQuery(qdrant_client, "Final_News_Articles_Collection")
    embedding = news_pipeline.embedder
    vector = embedding.get_embedding("Indian stock market news today: Sensex, Nifty, top gainers, top losers, Q3 results, earnings, stock movements, macroeconomic updates, sector performance")
    current_datetime = datetime.now()
    news_results = simple_news_query.query_news(vector, "2025-02-09", limit=10)

    llm_client = news_pipeline.metadata_extractor.client
    news_summarizer = NewsSummarizer(llm_client)
    summarized_news = news_summarizer.generate_summary(news_results)

    return {"summarized_news": summarized_news}

# Initialize Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(run_news_pipeline, "cron", hour=17, minute=0)  # Runs daily at 5 PM
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    """Gracefully shuts down the scheduler when FastAPI stops."""
    scheduler.shutdown()







# chatbot api

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI
app = FastAPI(title="Stock Market API", version="1.0")

# Initialize LLM Client
llm_client = Groq(api_key=GROQ_API_KEY)

# --- Function to Fetch Stock News and Generate Summary ---
def get_stock_news(symbol: str):
    url = f'https://eodhd.com/api/news?s={symbol}.US&offset=0&limit=10&api_token={EODHD_API_KEY}&fmt=json'
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Unable to fetch stock news.")
    
    return response.json()

# --- Function to Fetch Commodity Data ---
def get_commodity_data(function: str):
    url = f'https://www.alphavantage.co/query?function={function}&interval=daily&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Unable to fetch data for {function}.")
    
    return response.json()

# --- Function to Fetch Real GDP Data ---
def get_gdp_data():
    url = f'https://www.alphavantage.co/query?function=REAL_GDP&interval=annual&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Unable to fetch GDP data.")
    
    return response.json()

# --- Function to Fetch Crypto Exchange Rate ---
def get_crypto_exchange_rate(from_currency: str, to_currency: str):
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Unable to fetch exchange rate for {from_currency} to {to_currency}.")
    
    return response.json()

# --- Function to Fetch Global Stock Quote ---
def get_global_quote(symbol: str):
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Unable to fetch global quote for {symbol}.")
    
    return response.json()

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

# --- API Endpoints ---
class UserQuery(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Welcome to the Stock Market API"}

@app.post("/query")
def process_query(user_query: UserQuery):
    response = agent.chat(user_query.query)
    return {"response": response}

# Need to keep limits for the data that is fetched from the APIs
# Need to include the generalised question as a tool and we need to send that from the llm

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
