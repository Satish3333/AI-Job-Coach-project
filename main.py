import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables (if needed)
load_dotenv()

####
## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Initialize OpenAI embeddings model
embedding_model = OpenAIEmbeddings(openai_api_key="your_openai_api_key")

# Initialize FAISS index (for storing embeddings)
embedding_dimension = embedding_model.dimension  # Dimension of the embeddings
index = faiss.IndexFlatL2(embedding_dimension)

# Initialize Streamlit
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Load and embed documents when needed (Arxiv, Wikipedia)
def load_and_embed_documents():
    # Retrieve documents from Arxiv and Wikipedia
    papers = arxiv_wrapper.load_data(query="Machine Learning")
    wiki_articles = api_wrapper.load_data(query="Machine Learning")

    # Convert documents to embeddings
    paper_embeddings = embedding_model.embed_documents(papers)
    wiki_embeddings = embedding_model.embed_documents(wiki_articles)

    # Store embeddings in FAISS
    paper_embeddings_np = np.array(paper_embeddings)
    wiki_embeddings_np = np.array(wiki_embeddings)

    # Add to FAISS index
    index.add(paper_embeddings_np)
    index.add(wiki_embeddings_np)

# Load and embed documents once
load_and_embed_documents()

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Convert user query to embedding
    user_query_embedding = embedding_model.embed_query(prompt)

    # Perform a search in FAISS index
    D, I = index.search(np.array([user_query_embedding]), k=3)  # Get top 3 similar documents

    # Retrieve the most relevant documents based on search results
    top_papers = [arxiv_wrapper.load_data()[i] for i in I[0]]
    top_wiki_articles = [api_wrapper.load_data()[i] for i in I[0]]

    # Combine results and generate a response
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # Pass the retrieved documents to the agent for generating a response
        response = search_agent.run(st.session_state.messages + [{"role": "system", "content": str(top_papers + top_wiki_articles)}], callbacks=[st_cb])
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
