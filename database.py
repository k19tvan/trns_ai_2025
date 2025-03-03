from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode
from load_data import get_regulations, load_documents
import os

QDRANT_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Uw4X-9FJaBxRg_ZhiATvPMm4bXEqsbY8qrO1PJkv1e4"
QDRANT_ENDPOINT = "https://5e1c77f1-7f19-477c-babe-9872d61bc018.europe-west3-0.gcp.cloud.qdrant.io"

def main():
    document = load_documents()
    regulations = get_regulations(document)
    
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    qdrant = QdrantVectorStore.from_documents(
        regulations, embedding=embedding, 
        url = QDRANT_ENDPOINT,
        api_key = QDRANT_KEY, 
        collection_name="Regulations", 
        retrieval_mode=RetrievalMode.DENSE, 
    )

if __name__ == "__main__": main()
