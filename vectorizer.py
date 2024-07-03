import os
from langchain_community.document_loaders import FireCrawlLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma, Neo4jVector
from utils.commons import embeddings
from utils.logger import logger

# Load enviroment variable from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "vectorstore")

def create_vectorstore():
    """Crawl the website, split the text, create embeddings and store them in a vectorstore"""

    def _crawl_pages(urls):
        firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

        if not firecrawl_api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable not set")

        logger.info("Loading documents from FireCrawl...")
        friecrawl_docs = []
        for url in urls:
            firecrawl_loader = FireCrawlLoader(url=url, api_key=firecrawl_api_key, mode="scrape")
            docs = firecrawl_loader.load()
            friecrawl_docs.extend(docs)
        logger.info("Documents loaded successfully.")
        return friecrawl_docs
    
    urls = [
        "https://www.mckinsey.com/featured-insights",
        "https://www.bain.com/insights/",
        "https://www.mckinsey.com/quarterly/overview",
        "https://www.ft.com/us",
        "https://www.bloomberg.com/africa",
    ]

    # Crawl the pages
    firecrawl_docs = _crawl_pages(urls)
    
    # Convert metadata values to strings if they are lists
    for doc in firecrawl_docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    # Load the PDF
    logger.info("Loading PDF...")
    pdf_loader = PyPDFLoader(file_path=os.path.join(current_dir, "sources", "WORLDBANKREPORT2023.pdf"))
    pdf_docs = pdf_loader.load()

    # Add Source to metadate of PDF Document
    for doc in pdf_docs:
        doc.metadata["Source"] = "World Bank Report 2023"
    logger.info("PDF loaded successfully.")

    # Combine the documents
    logger.info("Combining documents...")
    docs = firecrawl_docs + pdf_docs
    logger.info(f"Documents combined successfully.\n\nTotal Documents: {len(docs)}\n\n{'-'*50}\n{docs[0]}\n{'-'*50}\n")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    # Create the vectorstore Chroma
    Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persistent_directory)
    logger.info("Chroma vector created successfully.")

    # Create Neo4j vector
    Neo4jVector.from_documents(splits, embeddings, url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USER"), password=os.getenv("NEO4J_PASSWORD"))
    logger.info("Neo4j vector created successfully.")
    return True

vectorstore = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def query_vectorstore(query):
    """Query the vectorstore with a question"""
    relevant_docs = retriever.invoke(query)
    print(relevant_docs)
    return relevant_docs