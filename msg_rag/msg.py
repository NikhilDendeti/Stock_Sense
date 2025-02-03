import os
import json
from crewai_tools import ScrapeWebsiteTool
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from groq import Groq
from tenacity import retry, wait_fixed, stop_after_attempt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API Keys
if not QDRANT_API_KEY or not QDRANT_URL or not GROQ_API_KEY:
    raise ValueError("Missing API keys. Make sure they are set in the .env file.")

# Web Scraping
tool = ScrapeWebsiteTool(website_url="https://www.indiatoday.in/business/story/sensex-crashes-700-points-why-is-the-indian-stock-market-falling-today-main-reasons-rupee-us-tariff-2673801-2025-02-03")
text = tool.run()
print(text)
text = text.replace('\u200b', ' ')

# Chunking with overlap
chunk_size = 750
overlap = 200
chunks = []
metadata_list = []

for i in range(0, len(text), chunk_size - overlap):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
    metadata_list.append({"source_url": tool.website_url, "chunk_index": i // (chunk_size - overlap)})

# Embedding Model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).tolist()

# Qdrant Configuration
collection_name = "web_scraped_data"
vector_size = 384
vector_params = VectorParams(size=vector_size, distance=Distance.COSINE)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
def check_or_create_collection(client, collection_name, vector_params):
    """Check if a collection exists in Qdrant; if not, create it."""
    existing_collections = client.get_collections()
    collection_names = [collection.name for collection in existing_collections.collections]
    if collection_name not in collection_names:
        client.create_collection(collection_name=collection_name, vectors_config=vector_params)
        print(f"Collection '{collection_name}' created successfully.")
    else:
        print(f"Collection '{collection_name}' already exists.")

check_or_create_collection(client, collection_name, vector_params)

# Groq Client for LLM Extraction
llm_client = Groq(api_key=GROQ_API_KEY)

def extract_metadata(chunk):
    """
    Use the Groq LLM to extract structured metadata from a text chunk.
    Returns a dict with keys: industries, stocks, date, news_type, sentiment, summary.
    """
    prompt = f"""
        Extract structured metadata from the following text and output only valid JSON without any additional text.

        Text:
        "{chunk}"

        The output must strictly follow this format:
        {{
        "industries": ["Industry1", "Industry2"],
        "stocks": ["Stock1", "Stock2"],
        "date": "YYYY-MM-DD",
        "news_type": ["Category1", "Category2"],
        "sentiment": "Positive/Negative/Neutral",
        "summary": "Concise summary of the text."
        }}
        """

    try:
        response = llm_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract metadata accurately."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=250,
            top_p=0.9
        )
    except Exception as e:
        print(f"LLM API call failed: {e}")
        return None

    extracted_text = response.choices[0].message.content.strip()
    print("LLM raw response:", extracted_text)
    
    try:
        extracted_metadata = json.loads(extracted_text)
        return extracted_metadata
    except json.JSONDecodeError:
        print("Error parsing LLM response. Skipping this chunk.")
        return None

# Store Extracted Metadata in Qdrant
points = []
for idx, (embedding, chunk, base_meta) in enumerate(zip(embeddings, chunks, metadata_list)):
    extracted_metadata = extract_metadata(chunk)
    if extracted_metadata:
        # Optionally merge the base metadata with the extracted metadata
        merged_metadata = {**base_meta, **extracted_metadata}
        points.append(PointStruct(id=idx, vector=embedding, payload=merged_metadata))

if points:
    client.upsert(collection_name=collection_name, points=points)
    print(f"Inserted {len(points)} records into Qdrant.")

# Query Example
query_text = "What is the impact of the 2025 budget on technology stocks?"
query_embedding = model.encode([query_text]).tolist()[0]

results = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5
)

print("\nRetrieved Results:")
for result in results:
    print(json.dumps(result.payload, indent=2))
