import os
from dotenv import load_dotenv
from utils.logger import logger
from langchain_community.graphs import Neo4jGraph


# Load environment variables
load_dotenv()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"), 
    username=os.getenv("NEO4J_USER"), 
    password=os.getenv("NEO4J_PASSWORD")
)

graph.query("MATCH (n) DETACH DELETE n")