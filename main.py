import streamlit as st
from dotenv import load_dotenv
import os
from langchain.document_loaders import PyMuPDFLoader, UnstructuredURLLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
import base64
import httpx
import re
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(page_title="RAG Flow Application", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #F5F7FA;
    }
    .main-container {
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #4A90E2;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #357ABD;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .user-bubble {
        background-color: #E1E9F0;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        align-self: flex-end;
    }
    .assistant-bubble {
        background-color: #fff;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        align-self: flex-start;
        border: 1px solid #E1E9F0;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🔍 Multimedia Retrieval-Augmented Generation (RAG) Framework for Unified Knowledge Extraction.")

# Sidebar for configurations
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    available_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini"]
    model_name = st.selectbox("🤖 Choose Model", available_models)
    
    if api_key and model_name:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
        st.success("✅ API key and model validated!")
    
    # File Uploaders
    st.subheader("📂 Data Sources")
    pdf_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
    web_urls = st.text_area("🌐 Enter Web URLs (comma separated)")
    youtube_videos = st.text_area("🎥 Enter YouTube URLs (comma separated)")
    image_urls = st.text_area("🖼️ Enter Image URLs (comma separated)")
    load_data_button = st.button("📥 Load Data")

# Function to load data from PDFs
def load_pdfs(files):
    documents = []
    for file in files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyMuPDFLoader(file.name)
        documents.extend(loader.load())
        os.remove(file.name)
    return documents

# Function to load data from URLs
def load_urls(urls):
    return UnstructuredURLLoader(urls=urls).load() if urls else []

# Function to load YouTube data
def load_youtube(video_urls):
    documents = []
    for video_url in video_urls:
        match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", video_url)
        if match:
            loader = YoutubeLoader(match.group(1))
            documents.extend(loader.load())
        else:
            st.error(f"Invalid YouTube URL: {video_url}")
    return documents

# Function to process images
def process_images(image_urls):
    documents = []
    for image_url in image_urls:
        try:
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            message = HumanMessage(content=[
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ])
            response = llm.invoke([message])
            documents.append(Document(page_content=response.content))
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    return documents

# Load and process data
if load_data_button:
    pdf_docs = load_pdfs(pdf_files) if pdf_files else []
    url_docs = load_urls(web_urls.split(',')) if web_urls else []
    youtube_docs = load_youtube(youtube_videos.split(',')) if youtube_videos else []
    image_docs = process_images(image_urls.split(',')) if image_urls else []
    all_documents = pdf_docs + url_docs + youtube_docs + image_docs

    if all_documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(all_documents)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        PROMPT_TEMPLATE = """
        You are an intelligent assistant designed for answering questions based on retrieved context.
Carefully read the provided context and use it exclusively to formulate your response.
If the context does not contain enough information to answer the question, explicitly state:
"I don't know based on the provided context."
Your responses should be clear, relevant, and limited to three sentences or fewer.
Do not include information beyond the given context.
        
        Context:
        {context}
        
        Question:
        {question}
        
        Answer:
        """
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
        
        st.session_state["qa_chain"] = qa_chain
        st.session_state["chat_history"] = []
        st.success("✅ Data loaded successfully! Start chatting below.")
    else:
        st.error("No valid data sources found. Please upload or enter URLs.")

# Chatbot Interface
if "qa_chain" in st.session_state:
    st.header("💬 Querymaster")
    user_query = st.text_input("Ask a question")
    if user_query:
        response = st.session_state["qa_chain"].invoke(user_query)
        st.session_state["chat_history"].append((user_query, response["result"]))
    for user, bot in st.session_state["chat_history"]:
        st.write(f"👤 {user}")
        st.write(f"🤖 {bot}")
