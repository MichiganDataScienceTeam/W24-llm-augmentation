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
st.session_state['openai_api_key'] = os.getenv("OPENAI_API_KEY")
openai_api_key = st.session_state.get('openai_api_key')
if not openai_api_key:
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

# Update the retriever when a new file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    file_path = save_uploaded_file(uploaded_file)

    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local('vectorstore/db_faiss')
    retriever = db.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "search_documents",
        "Searches and returns excerpts from additional documents provided by users.",
    )

    llm = ChatOpenAI(temperature=0)
    tools = [wiki_tool, retriever_tool]
    prompt = hub.pull("hwchase17/openai-tools-agent")
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

# Create containers for chat history and user input
response_container = st.container()
container = st.container()

# User input area
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Input", placeholder="Enter a message", key='input', label_visibility="collapsed")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state['history']
        })

        st.session_state['history'].append(HumanMessage(content=user_input))
        st.session_state['history'].append(AIMessage(result["output"]))

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(result["output"])

# Chat history
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