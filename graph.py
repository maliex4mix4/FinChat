import os
from dotenv import load_dotenv
from utils.logger import logger
from langchain_community.graphs import Neo4jGraph


# Load environment variables
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"), 
    username=os.getenv("NEO4J_USER"), 
    password=os.getenv("NEO4J_PASSWORD")
)

