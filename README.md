# Chatbot_With_Financial_Data

Savannah Assesment

The chatbot leverages OpenAI API for user query comprehension complemented with RAG for up-to-date information retrieval.

Tools Used: 
  1. LangChain framework: for user query synthesis and reply generation
  2. Flask: for web gui
  3. Groq (llama3-70b-8192): for language understanding
  4. Insights: for current information retrieval; 
     - [Link](https://www.mckinsey.com/featured-insights)
     - <https://www.bain.com/insights/>
     - <https://www.mckinsey.com/quarterly/overview>
     - <https://www.ft.com/us>


1. Download this clone repository
2. Create a virtual environment

     `python -m venv .venv`
3. Activate the virtual environment

    `source .venv/bin/activate` or `.venv/bin/activate` [for Windows]

4. Install the requirements

   `pip install -r requirements.txt`

5. Add your OpenAI API key in a .env file
6. On the terminal run the command below 

     `python app.py`
7. You can punch in the text: "What is Finacial times saying about US?"
