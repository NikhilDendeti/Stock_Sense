import http.client
import json
import uuid
from sentence_transformers import SentenceTransformer
from crewai_tools import ScrapeWebsiteTool
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchAny
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from groq import Groq
from datetime import datetime
import time

class NewsFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.conn = http.client.HTTPSConnection("google.serper.dev")
        self.headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

    def fetch_news(self, query="stock market news today india", gl="in", search_type="search"):
        payload = json.dumps({"q": query, "gl": gl, "type": search_type, "engine": "google"})
        self.conn.request("POST", "/search", payload, self.headers)
        res = self.conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))


class WebScraper:
    def __init__(self):
        self.tool = ScrapeWebsiteTool()

    def scrape_articles(self, all_articles):
        scraped_articles = []
        for i in all_articles:
            article = {}
            tool = ScrapeWebsiteTool(website_url=i["link"])
            scraped_text = tool.run()
            article["title"] = i["title"]
            article["link"] = i["link"]
            article["source"] = i["source"]
            article["date"] = i["date"]
            article["scraped_text"] = scraped_text
            scraped_articles.append(article)

        return scraped_articles

class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)


class QdrantHandler:
    def __init__(self, api_key, url, collection_name, vector_size=384):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._initialize_collection()

    def _initialize_collection(self):
        try:
            existing_collections = self.client.get_collections()
            collection_names = [collection.name for collection in existing_collections.collections]
            if self.collection_name not in collection_names:
                vector_params = VectorParams(size=self.vector_size, distance=Distance.COSINE)
                self.client.create_collection(collection_name=self.collection_name, vectors_config=vector_params)
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error initializing Qdrant collection: {e}")

    def upsert_points(self, points, batch_size=100):
        try:
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch)
                print(f"✅ Successfully upserted {len(batch)} chunks into Qdrant.")
            print(f"🎯 Total {len(points)} chunks inserted successfully!")
        except Exception as e:
            print(f"❌ Error while upserting to Qdrant: {e}")


class MetadataExtractor:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def clean_json_output(self, response_text):
        """
        Cleans the JSON output by removing any leading/trailing code block markers
        and ensuring it's properly formatted for parsing.
        """
        response_text = response_text.strip().strip("```json").strip("```").strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            print("Error parsing LLM response. Returning None.")
            return None

    def extract_metadata(self, text):
        prompt = f"""
        Extract structured metadata from the following text and return in JSON format.
        Ensure that:
        - "industries" includes relevant sectors from a predefined list but allows emerging industries.
        - "news_type" is categorized using predefined categories but allows flexibility.
        - "stocks" should contain at most 5 stock tickers.
        - "sentiment" is categorized as Positive, Negative, or Neutral.
        - Don't include JSON in the response heading.
        -Don't include any headings in the response body. Like metadata:. I only want Pure JSON format.

        Predefined Industries:
        ["Technology", "Healthcare", "Finance", "Energy", "Consumer Goods", "Real Estate", "Industrials", "Emerging Sectors"]

        Predefined News Types:
        ["Earnings", "Stock Movements", "Mergers & Acquisitions", "Regulatory & Legal", "Macroeconomic",
        "Company Announcements", "Market Trends", "Geopolitical Impact"]
        
        Text:
        "{text}"

        Format:
        {{
          "industries": ["Industry1", "Industry2"],
          "stocks": ["Stock1", "Stock2", "Stock3", "Stock4", "Stock5"],
          "date": "YYYY-MM-DD",
          "news_type": ["Category1", "Category2"],
          "sentiment": "Positive/Negative/Neutral",
          "summary": "Concise summary of the text."
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "Extract metadata accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_completion_tokens=250,
                top_p=0.9
            )
            extracted_text = response.choices[0].message.content.strip()
            # print(extracted_text)
            extracted_metadata = json.loads(extracted_text)
            print(extracted_metadata)
            return self.clean_json_output(extracted_text)
        except json.JSONDecodeError:
            print("Error parsing LLM response. Skipping this chunk.")
            return None


        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract metadata accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=250,
            top_p=0.9
        )
        return json.loads(response.choices[0].message.content.strip())


class NewsPipeline:
    def __init__(self, news_api_key, qdrant_api_key, qdrant_url, collection_name, llm_api_key):
        self.fetcher = NewsFetcher(news_api_key)
        self.scraper = WebScraper()
        self.embedder = EmbeddingModel()
        self.qdrant = QdrantHandler(qdrant_api_key, qdrant_url, collection_name)
        self.metadata_extractor = MetadataExtractor(llm_api_key)
        self.all_scraped_articles = []

    def process_news(self):
        news_data = self.fetcher.fetch_news()
        articles = news_data.get("topStories", [])
        
        # Formatting the extracted data, including sitelinks if available

        scraped_articles_with_sitelinks = []

        for entry in news_data["organic"]:
            source = entry["link"].split("/")[2]

            if "sitelinks" in entry:
                for sitelink in entry["sitelinks"]:
                    article = {
                        "title": entry["title"],
                        "link": sitelink["link"],
                        "date": entry.get("date", "N/A"),
                        "source": source,
                        "scraped_text": "" 
                    }
                    scraped_articles_with_sitelinks.append(article)
            else:
                article = {
                    "title": entry["title"],
                    "link": entry["link"],
                    "date": entry.get("date", "N/A"),
                    "source": source, # Placeholder for now, as scraping is not required
                }
                scraped_articles_with_sitelinks.append(article)

        articles.extend(scraped_articles_with_sitelinks)
        
        print(len(articles), "Total articles found")
        scraped_articles = self.scraper.scrape_articles(articles)
        self.all_scraped_articles = scraped_articles
        return scraped_articles
    
    def process_scraped_articles(self, scraped_articles):
        all_points = []
        print(len(scraped_articles), "Total articles to process")
        for article in scraped_articles:
            node_parser = SentenceSplitter(chunk_size=350, chunk_overlap=50)
            document = Document(text=article["scraped_text"])
            nodes = node_parser.get_nodes_from_documents([document], show_progress=False)
            title_embedding = self.embedder.get_embedding(article["title"])
            print(len(nodes), "No of nodes found")
            for node in nodes:
                similarity = self.embedder.model.similarity(title_embedding, self.embedder.get_embedding(node.text))
                if similarity > 0.1:
                    current_datetime = datetime.now()
                    metadata = self.metadata_extractor.extract_metadata(node.text)
                    all_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=self.embedder.get_embedding(node.text),
                            payload={
                                "chunk": node.text,
                                "title": article["title"],
                                "industries": metadata.get("industries", []),
                                "stocks": metadata.get("stocks", [])[:5],
                                "date": current_datetime.strftime("%Y-%m-%d"),
                                "news_type": metadata.get("news_type", []),
                                "sentiment": metadata.get("sentiment", "Neutral"),
                                "summary": metadata.get("summary", ""),
                                "link": article["link"],
                                "source": article.get("source", "")
                            }
                        )
                    )
        print(len(all_points), "Total points found")
        self.qdrant.upsert_points(all_points)
        return all_points
        


class UserPreferences:
    def __init__(self, industries=None, stocks=None, sentiment=None, news_types=None, date=None):
        self.industries = industries if industries else []
        self.stocks = stocks if stocks else []
        self.sentiment = sentiment if sentiment else None
        self.news_types = news_types if news_types else []
        self.date = date if date else None

    def to_filter(self):
        conditions = []
        if self.industries:
            conditions.append(FieldCondition(key="industries", match=MatchAny(any=self.industries)))
        if self.stocks:
            conditions.append(FieldCondition(key="stocks", match=MatchAny(any=self.stocks)))
        if self.sentiment:
            conditions.append(FieldCondition(key="sentiment", match=MatchValue(value=self.sentiment)))
        if self.news_types:
            conditions.append(FieldCondition(key="news_type", match=MatchAny(any=self.news_types)))
        if self.date:
            conditions.append(FieldCondition(key="date", match=MatchValue(value=self.date)))
        return Filter(must=conditions) if conditions else None

class NewsQueryHandler:
    def __init__(self, qdrant_client, collection_name):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def get_news_for_user(self, user_preferences):
        query_filter = user_preferences.to_filter()
        response = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=10
        )
        return [point.payload for point in response[0]]


class NewsSummarizer:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_summary(self, news_articles):
        print(news_articles)
        if not news_articles:
            return "No relevant stock market news found today based on your preferences."
        news_text = "\n".join([f"- {article['summary']} ({article['link']})" for article in news_articles])
        prompt = f"""
📊 Market Overview:
Start with a clear sentiment statement (bullish, bearish, volatile, mixed)
Mention key market drivers (global trends, sector movements, institutional flows, economic data, corporate actions)
Summarize major index movements (Sensex, Nifty50, BSE, Nasdaq, Dow Jones, etc.) with percentage changes
Highlight any major trends or sector rotations driving market sentiment

📌 Top Headlines & Key Developments:
Cover all major market-moving news, including:
- Biggest gainers & losers with reasons for movement
- Earnings reports (YoY & QoQ performance, revenue & profit changes)
- FII/DII activity, block deals, insider trades
- Government policies & regulatory updates affecting sectors
- Global market cues influencing domestic trends
- Sector-specific news (IT, Pharma, Banking, Energy, etc.)

📉 Indices & Sectoral Performance:
Summarize major index movements (Nifty50, Sensex, Bank Nifty, Midcap, Smallcap) with absolute & percentage changes
Breakdown of sector-wise performance (Top gaining & losing sectors)
FII/DII net inflow-outflow details (with ₹ figures)
Mention any significant institutional moves that impacted sectors

🌎 Global & Macro Factors:
Cover global market movements (Dow Jones, S&P 500, FTSE, Nikkei, etc.)
Track USD-INR movement, crude oil price fluctuations, bond yields
Mention important macroeconomic indicators (GDP growth, inflation, PMI, IIP, interest rate policies)

💡 Final Takeaway & Market Outlook:
Provide a forward-looking view on what may influence the market next
Mention any upcoming economic events, data releases, or geopolitical risks
Highlight sectoral trends & potential stock movements for the next session

-If you don't have any data points mentioned above just don't hallucinate and give random answers. Don't answer that question.
-Don't include any emojis. Bold the sideheadings only.
📰 **Actual News Articles:**  
{news_text}
"""
        response = self.llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()



class SimpleNewsQuery:
    def __init__(self, qdrant_client, collection_name):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

    def query_news(self, search_query, date=None, limit=10):
        """
        Queries Qdrant for news articles using a simple text-based search.
        """
        query_filter = None
        if date:
            query_filter = Filter(must=[FieldCondition(key="date", match=MatchValue(value=date))])

        response = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=search_query,
            query_filter=query_filter,
            limit=limit
        )
        return [point.payload for point in response]

# qdrant_client = QdrantClient(url="https://3f0e0b2d-447a-48bd-8c2f-3a227ff85295.eu-west-1-0.aws.cloud.qdrant.io:6333/", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2MTc3MDAyfQ.v-Kra-ZRUdTAe0OBCmCjEvZi8rW_HKY0Cw2Vp-AM5g4", timeout=60)
# simple_news_query = SimpleNewsQuery(qdrant_client, "stock_market_news_india_5")
# embedding = EmbeddingModel()
# vector = embedding.get_embedding("Indian stock market news today: Sensex, Nifty, top gainers, top losers, Q3 results, earnings, stock movements, macroeconomic updates, sector performance")
# current_datetime = datetime.now()
# news_results = simple_news_query.query_news(vector, current_datetime.strftime("%Y-%m-%d"), limit=5)

# llm_client = Groq(api_key="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")
# news_summarizer = NewsSummarizer(llm_client)
# print(f"📢 Queried News Articles:\n{news_results}")

# summarized_news = news_summarizer.generate_summary(news_results)

# print(f"📢 Stock Market Summary:\n{summarized_news}")



# Usage Example
# user_prefs = UserPreferences(
#     industries=["Technology", "Finance"],
#     stocks=[],
#     sentiment="Negative",
#     news_types=["Macroeconomic", "Market Trends", "Geopolitical Impact"]
# )


# news_query_handler = NewsQueryHandler(qdrant_client, "stock_market_news_india_5")
# llm_client = Groq(api_key="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")
# news_summarizer = NewsSummarizer(llm_client)
# news_articles = news_query_handler.get_news_for_user(user_prefs)
# summarized_news = news_summarizer.generate_summary(news_articles)

# print(f"📢 Stock Market Summary:\n{summarized_news}")

# user_prefs = UserPreferences(
#     industries=["Technology", "Finance"],
#     stocks=["AAPL", "TSLA"],
#     sentiment="Positive",
#     news_types=["Earnings", "Stock Movements"]
# )

# news_query_handler = NewsQueryHandler(qdrant_client, "stock_market_news")
# llm_client = Groq(api_key="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")
# news_summarizer = NewsSummarizer(llm_client)