from fastapi import FastAPI, BackgroundTasks, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import os
import logging
import requests
from dotenv import load_dotenv
from news_processor import NewsPipeline, SimpleNewsQuery, NewsSummarizer, EmbeddingModel, QdrantClient
from groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq as GroqLLM
from llama_index.core.tools import FunctionTool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
import re

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

def get_stock_news(symbol: str):
    try:
        response = requests.get(
            f'https://eodhd.com/api/news?s={symbol}.US',
            params={'api_token': EODHD_API_KEY}
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Stock news API error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch stock news")

def get_commodity_data(function: str):
    try:
        response = requests.get(
            f'https://www.alphavantage.co/query?function={function}&apikey={ALPHA_VANTAGE_API_KEY}'
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Commodity data API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch {function} data")

try:
    agent = ReActAgent.from_tools(
        [FunctionTool.from_defaults(get_stock_news), FunctionTool.from_defaults(get_commodity_data)],
        llm=GroqLLM(model="llama3-70b-8192", api_key=GROQ_API_KEY),
        verbose=True
    )
except Exception as e:
    logging.error(f"Failed to initialize ReAct Agent: {str(e)}")
    agent = None

class UserQuery(BaseModel):
    query: str

@app.post("/query")
async def process_query(user_query: UserQuery):
    if not agent:
        raise HTTPException(status_code=500, detail="Chat service unavailable")
    
    try:
        response = agent.chat(user_query.query)
        return {"response": str(response)}
    except Exception as e:
        logging.error(f"Query processing failed: {str(e)}")
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
