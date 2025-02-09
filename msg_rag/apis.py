from fastapi import FastAPI, BackgroundTasks
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime
from news_processor import NewsPipeline, UserPreferences, NewsQueryHandler, NewsSummarizer, SimpleNewsQuery
from qdrant_client import QdrantClient

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
