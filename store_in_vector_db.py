from pinecone import Pinecone
import pinecone
from sentence_transformers import SentenceTransformer
import uuid
import logging
from typing import List, Dict
import os
import json

dataset_path = "dataset/json"
datasets = [os.path.join(dataset_path, i) for i in os.listdir(dataset_path) if i.endswith(".json")]
all_chunks = []
for fpath in datasets:
    with open(fpath) as f:
        all_chunks.extend(json.load(f))

pc = Pinecone(api_key=os.getenv("pincone_api"), host = os.getenv("pincone_host"))

def create_index_if_not_exists(index_name: str, dimension: int):
    # Check if the index exists and create it if not
    if index_name not in pc.list_indexes():
        print(f"Index '{index_name}' not found. Creating index...")
        pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")



def upsert_to_pinecone(
    chunks: List[Dict],

    index_name: str = "financial-rag",
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> None:
    """
    Embeds and upserts chunked data to Pinecone with metadata.

    Args:
        chunks (List[Dict]): List of dicts with keys like 'company', 'year', 'page', 'summary', and 'clean_content'.
        pinecone_api_key (str): Your Pinecone API key.
        pinecone_env (str): Your Pinecone environment region (e.g., "gcp-starter").
        index_name (str): Name of the Pinecone index to use (default: "financial-rag").
        embedding_model_name (str): Sentence transformer model to use for embedding (default: MiniLM).

    Returns:
        None
    """

    try:
        print(pc.list_indexes())
        # Load index
            # First, ensure the index exists
        create_index_if_not_exists(index_name, dimension=348)  # Match the dimension of the embedding model


        index = pc.Index(index_name)

        # Load model
        model = SentenceTransformer(embedding_model_name)

        vectors = []
        for chunk in tqdm(chunks):
            try:
                vector = model.encode(chunk["summarized"]).tolist()

                metadata = {
                    "company": chunk.get("company", chunk['company']),
                    "year": chunk.get("year", chunk['year']),
                    "page": chunk.get("page",chunk['page_num']),
                    "summarized": chunk.get("summarized", ""),
                    "content": chunk.get("clean_content", "")
                }

                vectors.append((str(uuid.uuid4()), vector, metadata))
            except Exception as e:
                print(f"Failed to process chunk on page {chunk.get('page', '?')}: {e}")

        # Batch upsert (in chunks of 100 for large datasets)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted {i + len(batch)}/{len(vectors)} vectors into Pinecone")

        logging.info("✅ All vectors successfully upserted into Pinecone.")

    except Exception as e:
        print(f"❌ Pinecone upsert failed: {e}")
        raise e

upsert_to_pinecone(all_chunks)