import http.client
import json
import uuid
from sentence_transformers import SentenceTransformer
from crewai_tools import ScrapeWebsiteTool
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from groq import Groq


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
        - "summary" is concise (under 25 words).
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
            print(extracted_text)
            extracted_metadata = json.loads(extracted_text)
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

    def process_news(self):
        news_data = self.fetcher.fetch_news()
        articles = news_data.get("topStories", [])
        scraped_articles = self.scraper.scrape_articles(articles)
        all_points = []
        print(len(scraped_articles))
        for article in scraped_articles[3:4]:
            node_parser = SentenceSplitter(chunk_size=350, chunk_overlap=50)
            document = Document(text=article["scraped_text"])
            nodes = node_parser.get_nodes_from_documents([document], show_progress=False)
            title_embedding = self.embedder.get_embedding(article["title"])

            for node in nodes:
                similarity = self.embedder.model.similarity(title_embedding, self.embedder.get_embedding(node.text))
                if similarity > 0.1:
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
                                "date": metadata.get("date", ""),
                                "news_type": metadata.get("news_type", []),
                                "sentiment": metadata.get("sentiment", "Neutral"),
                                "summary": metadata.get("summary", ""),
                                "link": article["link"],
                                "source": article.get("source", "")
                            }
                        )
                    )

        self.qdrant.upsert_points(all_points)


# Initialize and run the pipeline
news_pipeline = NewsPipeline(
    news_api_key="7549ec8c3f790b338e0e57e8f5014c1ac1782714",
    qdrant_api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2MTc3MDAyfQ.v-Kra-ZRUdTAe0OBCmCjEvZi8rW_HKY0Cw2Vp-AM5g4",
    qdrant_url="https://3f0e0b2d-447a-48bd-8c2f-3a227ff85295.eu-west-1-0.aws.cloud.qdrant.io:6333/",
    collection_name="stock_market_news_india_5",
    llm_api_key="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"
)
news_pipeline.process_news()
