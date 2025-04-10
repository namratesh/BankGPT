from  pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
pc = Pinecone(api_key=os.getenv("pincone_api"), host = os.getenv("pincone_host"))
def query_pinecone(
    query: str,
    index_name: str,
    top_k: int = 5,
    filter_by: Dict = None,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Query Pinecone vector DB using semantic similarity and optional filters.

    Args:
        query (str): User's natural language question.
        api_key (str): Pinecone API key.
        host (str): Pinecone serverless host URL.
        index_name (str): Pinecone index name.
        top_k (int): Number of results to retrieve.
        filter_by (Dict): Optional metadata filter (e.g., {"company": "HDFC", "year": "2023-2024"}).
        embedding_model (str): SentenceTransformer model to use.

    Returns:
        List[Dict]: Retrieved results with metadata and similarity score.
    """

    try:
        # Initialize Pinecone
        
        index = pinecone.Index(index_name)

        # Load embedding model
        model = SentenceTransformer(embedding_model)
        query_vector = model.encode(query).tolist()

        # Build query payload
        query_args = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        if filter_by:
            query_args["filter"] = filter_by

        # Execute query
        results = index.query(**query_args)

        return results.get("matches", [])

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


from  pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
pc = Pinecone(api_key=os.getenv("pincone_api"), host = os.getenv("pincone_host"))
def query_pinecone(
    query: str,
    index_name: str = "financial-rag",
    top_k: int = 5,
    filter_by: Dict = None,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> List[Dict]:
    """
    Query Pinecone vector DB using semantic similarity and optional filters.

    Args:
        query (str): User's natural language question.
        api_key (str): Pinecone API key.
        host (str): Pinecone serverless host URL.
        index_name (str): Pinecone index name.
        top_k (int): Number of results to retrieve.
        filter_by (Dict): Optional metadata filter (e.g., {"company": "HDFC", "year": "2023-2024"}).
        embedding_model (str): SentenceTransformer model to use.

    Returns:
        List[Dict]: Retrieved results with metadata and similarity score.
    """

    try:
        # Initialize Pinecone
        
        index = pc.Index(index_name)

        # Load embedding model
        model = SentenceTransformer(embedding_model)
        query_vector = model.encode(query).tolist()

        # Build query payload
        query_args = {
            "vector": query_vector,
            "top_k": top_k,
            "include_metadata": True
        }
        if filter_by:
            query_args["filter"] = filter_by

        # Execute query
        results = index.query(**query_args)

        return results.get("matches", [])

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return []


query = "What is SBI’s capital adequacy ratio in 2023?"
data = query_pinecone(query=query)
for r in data:
    print(f"\n✅ Match | Score: {r['score']:.3f}")
    print(f"Company: {r['metadata']['company']}, Year: {r['metadata']['year']}, Page: {r['metadata']['page']}")
    print(f"Snippet:\n{r['metadata']['content'][:500]}...")