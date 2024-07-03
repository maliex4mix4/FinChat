import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()


# Configure WebBaseLoader to fetch documents
loader = WebBaseLoader(web_paths=("https://www.mckinsey.com/featured-insights", "https://www.bain.com/insights/", "https://www.mckinsey.com/quarterly/overview", "https://www.ft.com/us"))
docs = loader.load()

# Split documents into chunks for processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=FastEmbedEmbeddings(), persist_directory="./vectorstore")

def response(user_query):

    # Set up the retriever for information retrieval
    retriever = vectorstore.as_retriever(search_type="similarity")
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

    def format_docs(docs):
        return [doc.page_content for doc in docs]

    # Define the RAG prompt template
    template = """Use the provided context to formulate your response to the question below. If unsure, explicitly state that you don't have the information. Avoid speculation; provide precise and concise answers.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:"""


    # Create a custom RAG prompt
    custom_rag_prompt = PromptTemplate.from_template(template)

    # Define the RAG chain for processing the query and context
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the RAG chain with the user query and return the result
    return rag_chain.invoke(user_query)