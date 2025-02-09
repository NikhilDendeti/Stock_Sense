import json
import uuid
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from crewai_tools import ScrapeWebsiteTool

class StockDataScraper:
    def __init__(self, url):
        self.url = url
        self.scraped_text = ""
    
    def scrape(self):
        tool = ScrapeWebsiteTool(website_url=self.url)
        self.scraped_text = tool.run()
        return self.scraped_text

class DataExtractor:
    def __init__(self, api_key):
        self.llm_client = Groq(api_key=api_key)
        self.company_format = { 
            "Company": "[Company Name]",
            "Stock Info": {
                "BSE Code": "[BSE Code]",
                "NSE Code": "[NSE Code]",
                "Current Price": "[Current Price]",
                "Market Cap (Cr)": "[Market Cap in Crores]",
                "High/Low": ["[High Price]", "[Low Price]"],
                "Stock P/E": "[Stock P/E ratio]",
                "Book Value (₹)": "[Book Value]",
                "Dividend Yield (%)": "[Dividend Yield]",
                "ROCE (%)": "[Return on Capital Employed]",
                "ROE (%)": "[Return on Equity]",
                "Face Value (₹)": "[Face Value]"
            },
            "Financials": {},
            "Ratios": {},
            "Shareholding Pattern": {},
            "Announcements": []
        }
    
    def extract(self, scraped_text):
        prompt = f"""
          - Convert the following text into JSON format using this template:
          {self.company_format}
          
          - Text:
          {scraped_text}
        """
        response = self.llm_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Extract metadata accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.9
        )
        
        extracted_text = response.choices[0].message.content
        try:
            extracted_text = extracted_text.replace("'", '"')
            return json.loads(extracted_text)
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
            return None

class QdrantDatabase:
    def __init__(self, api_key, url, collection_name, vector_size=384):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.ensure_collection()

    def ensure_collection(self):
        try:
            existing_collections = self.client.get_collections()
            collection_names = [collection.name for collection in existing_collections.collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Error occurred while checking/creating collection: {e}")
    
    def insert_data(self, company_data, model):
        points = []
        section_names = ["Stock Info", "Financials", "Ratios", "Shareholding Pattern", "Announcements"]

        for section in section_names:
            if section in company_data:
                text_data = f"{section}: {company_data[section]}"
                vector = model.encode(text_data).tolist()
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"Company": company_data["Company"], "Section": section, "Data": company_data[section]}
                ))
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_text, model):
        query_vector = model.encode(query_text).tolist()
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=2
        )
        return [point.payload for point in search_result] if search_result else "No relevant data found."

class StockInfoChatbot:
    def __init__(self, llm_client, db):
        self.llm_client = llm_client
        self.db = db
    
    def get_response(self, query_text, model):
        context = self.db.query(query_text, model)
        prompt = f"""
          Based on the {context}, answer the following question:
          {query_text}
        """
        response = self.llm_client.chat.completions.create(
            model="llama-3.3-70b-specdec",
            messages=[
                {"role": "system", "content": "Extract metadata accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()

# Example Usage
if __name__ == "__main__":
    SCRAPER_URL = "https://www.screener.in/company/INFY/consolidated/"
    GROQ_API_KEY = "your_groq_api_key"
    QDRANT_API_KEY = "your_qdrant_api_key"
    QDRANT_URL = "your_qdrant_url"
    COLLECTION_NAME = "company_info"
    
    scraper = StockDataScraper(SCRAPER_URL)
    scraped_data = scraper.scrape()
    
    extractor = DataExtractor(GROQ_API_KEY)
    company_data = extractor.extract(scraped_data)
    
    db = QdrantDatabase(QDRANT_API_KEY, QDRANT_URL, COLLECTION_NAME)
    db.insert_data(company_data, model)
    
    chatbot = StockInfoChatbot(extractor.llm_client, db)
    response = chatbot.get_response("Infosys latest results?", model)
    print(response)