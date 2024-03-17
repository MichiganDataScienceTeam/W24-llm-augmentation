# To run this example, make sure you run this in command line:
# pip install -r requirements.txt
# Then run the streamlit with this command:
# streamlit run streamlit-example.py
# (make sure you get the path to this .py file correct)

import os
import shutil
import streamlit as st
import tempfile
from dotenv import load_dotenv
from streamlit_chat import message

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, create_openai_tools_agent
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage

# Store OpenAI API key
load_dotenv()
st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY") # Get the api key from .env file and store it in a session variable
if not st.session_state['openai_api_key']: # Check the api key exists
    st.error("OpenAI API key not found")
    st.stop()

# Title
st.title("AI Chat With Documents")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey"]

# Function to save uploaded file to a temporary directory
def save_uploaded_file(uploaded_file):
    # Create a new temporary directory for each file/session
    if 'file_paths' not in st.session_state:
        st.session_state['file_paths'] = {}

    # Generate a temporary file name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        # Write the uploaded file's contents to the temporary file
        shutil.copyfileobj(uploaded_file, tmp_file)
        st.session_state['file_paths'][uploaded_file.name] = tmp_file.name

    return st.session_state['file_paths'][uploaded_file.name]

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Upload File", type="pdf")

# Create a Wikipedia search tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Initialize a LangChain chat agent
llm = ChatOpenAI(temperature=0)
tools = [wiki_tool]
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create a new agent with a new retriever when a new file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    file_path = save_uploaded_file(uploaded_file)

    # Load the pdf document
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    # Split and embed the text in the documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    # Store the embeddings in a database as local files
    db = FAISS.from_documents(texts, embeddings)
    db.save_local('vectorstore/db_faiss')
    retriever = db.as_retriever()

    # Create a tool for the agent that retrieves text from the database
    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        "Searches and returns excerpts from additional documents provided by users.",
    )

    # Initialize the new sagent
    llm = ChatOpenAI(temperature=0)
    tools = [wiki_tool, retriever_tool]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create streamlit containers for chat history and user input
response_container = st.container()
container = st.container()

# Display user input area
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Input", placeholder="Enter a message", key='input', label_visibility="collapsed")
        submit_button = st.form_submit_button(label='Send')

    # This runs when user enters message to chat bot
    if submit_button and user_input:
        # Get a response from the llm, by giving the user message and chat history to the agent
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state['history']
        })

        # Add the user message and llm response to the chat history
        st.session_state['history'].append(HumanMessage(content=user_input))
        st.session_state['history'].append(AIMessage(result["output"]))

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result["output"])

# Dislplay chat history
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

# At the end of the session or when you're done with the file, clean up the temporary files
def clean_up_files():
    if 'file_paths' in st.session_state:
        for file_path in st.session_state['file_paths'].values():
            if os.path.exists(file_path):
                os.remove(file_path)