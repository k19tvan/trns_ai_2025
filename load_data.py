from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_PATH = './data'

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len,
        add_start_index=True,
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_regulations(document):
    document = load_documents()[0].page_content
    docs = document.split("\n\n")
    return [Document(page_content=doc) for doc in docs]

print(len(get_regulations(load_documents())))

    