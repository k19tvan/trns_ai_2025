from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
import os
from qdrant_client import qdrant_client

QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Uw4X-9FJaBxRg_ZhiATvPMm4bXEqsbY8qrO1PJkv1e4"
QDRANT_ENDPOINT = "https://5e1c77f1-7f19-477c-babe-9872d61bc018.europe-west3-0.gcp.cloud.qdrant.io"

def get_retrievals(question, k = 2):

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    client = qdrant_client.QdrantClient(
        url=QDRANT_ENDPOINT,
        api_key=QDRANT_KEY,
    )
    
    qdrant = QdrantVectorStore(
        embedding=embedding,
        client=client,
        collection_name="Regulations",
        retrieval_mode=RetrievalMode.DENSE
    )
    
    return qdrant.similarity_search(question, k)





        
    