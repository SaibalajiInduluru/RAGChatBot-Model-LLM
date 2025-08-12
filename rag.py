import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import openai
import tempfile
import os
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS with glassmorphism, animations, and dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for dark theme */
    :root {
        --bg-primary: #0f0f23;
        --bg-secondary: #1a1a2e;
        --bg-glass: rgba(255, 255, 255, 0.05);
        --text-primary: #ffffff;
        --text-secondary: #a0a0a0;
        --accent: #6366f1;
        --accent-hover: #5b59f7;
        --success: #10b981;
        --error: #ef4444;
        --warning: #f59e0b;
        --send-button: #22c55e;
        --send-button-hover: #16a34a;
    }
    
    /* Global styles */
    .main, .block-container {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        color: var(--text-primary) !important;
        padding-top: 1rem !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.2) 0%, transparent 50%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        position: relative;
        z-index: 1;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .stSidebar .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSidebar {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSidebar > div {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
    }
    
    .stSidebar .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3) !important;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: white !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Chat container */
    .chat-container {
        background: var(--bg-glass);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
        position: relative;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 10px;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 8px 25px;
        margin: 1rem 0;
        margin-left: 15%;
        position: relative;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 0;
        margin-right: 15%;
        position: relative;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
        animation: slideInLeft 0.3s ease-out;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-50px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Status cards */
    .status-card {
        padding: 1rem 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid;
        backdrop-filter: blur(10px);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success);
        color: var(--success);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--error);
        color: var(--error);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        border-color: var(--warning);
        color: var(--warning);
    }
    
    @keyframes fadeInUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    /* Custom buttons - General styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        width: 100% !important;
        height: 3rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
        background: linear-gradient(135deg, #5b59f7 0%, #7c3aed 100%) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* ENHANCED SEND BUTTON STYLING - MAXIMUM VISIBILITY */
    .stForm .stButton > button,
    .stForm button[kind="formSubmit"],
    button[kind="formSubmit"],
    [data-testid="baseButton-secondary"],
    .stForm [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, var(--send-button) 0%, #16a34a 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        height: 3.5rem !important;
        min-height: 3.5rem !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5) !important;
        border: 2px solid rgba(34, 197, 94, 0.8) !important;
        border-radius: 15px !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        backdrop-filter: blur(10px) !important;
        position: relative !important;
        z-index: 10 !important;
        transition: all 0.2s ease !important;
    }
    
    .stForm .stButton > button:hover,
    .stForm button[kind="formSubmit"]:hover,
    button[kind="formSubmit"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    .stForm [data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, var(--send-button-hover) 0%, #15803d 100%) !important;
        box-shadow: 0 8px 30px rgba(34, 197, 94, 0.7) !important;
        transform: translateY(-3px) scale(1.02) !important;
        border-color: rgba(34, 197, 94, 1) !important;
    }
    
    .stForm .stButton > button:active,
    .stForm button[kind="formSubmit"]:active,
    button[kind="formSubmit"]:active,
    [data-testid="baseButton-secondary"]:active,
    .stForm [data-testid="baseButton-secondary"]:active {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.6) !important;
    }
    
    /* Ensure send button container is visible */
    .stForm > div > div {
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Input fields - ENHANCED VERSION FOR MAXIMUM VISIBILITY */
    .stTextInput > div > div > input, 
    .stTextInput input,
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stTextInput"] input,
    .stTextInput > label + div > div > input,
    div[data-testid="stTextInput"] input {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 2px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        color: #1a1a2e !important;
        backdrop-filter: blur(10px) !important;
        padding: 1rem 1.5rem !important;
        font-size: 1.1rem !important;
        height: 3.5rem !important;
        box-sizing: border-box !important;
        width: 100% !important;
        min-height: 3.5rem !important;
        font-weight: 500 !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextInput input:focus,
    [data-testid="stTextInput"] > div > div > input:focus,
    [data-testid="stTextInput"] input:focus,
    .stTextInput > label + div > div > input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3), 0 4px 20px rgba(99, 102, 241, 0.2) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transform: translateY(-1px) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextInput input::placeholder,
    [data-testid="stTextInput"] > div > div > input::placeholder,
    [data-testid="stTextInput"] input::placeholder,
    .stTextInput > label + div > div > input::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(26, 26, 46, 0.6) !important;
        font-weight: 400 !important;
    }
    
    /* Text input container styling */
    .stTextInput > div,
    [data-testid="stTextInput"] > div,
    .stTextInput > label + div {
        width: 100% !important;
        position: relative !important;
    }
    
    .stTextInput > div > div,
    [data-testid="stTextInput"] > div > div,
    .stTextInput > label + div > div {
        width: 100% !important;
        position: relative !important;
    }
    
    /* Form container styling to ensure proper alignment */
    .stForm {
        width: 100% !important;
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    /* Column styling for input form - Ensure button column is visible */
    [data-testid="column"] {
        width: 100% !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Specific styling for the button column */
    [data-testid="column"]:last-child {
        display: flex !important;
        align-items: flex-end !important;
        justify-content: center !important;
        min-height: 3.5rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Input container in form */
    .chat-input-container {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(15px) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        width: 100% !important;
    }
    
    /* Input section title */
    .input-section-title {
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stSelectbox > div > div > select {
        background: var(--bg-glass) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.3) !important;
    }
    
    /* Sidebar text inputs */
    .stSidebar .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 2px solid rgba(255, 255, 255, 0.35) !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    .stSidebar .stSelectbox > div > div > select {
        background: rgba(255, 255, 255, 0.12) !important;
        border: 2px solid rgba(255, 255, 255, 0.35) !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(99, 102, 241, 0.5) !important;
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent) !important;
        background: rgba(99, 102, 241, 0.1) !important;
    }
    
    .stFileUploader > div {
        color: var(--text-primary) !important;
    }
    
    .stFileUploader label {
        color: white !important;
    }
    
    /* File uploader drag and drop area */
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    }
    
    /* Floating elements */
    .floating-element {
        position: fixed;
        pointer-events: none;
        opacity: 0.1;
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(5deg); }
        66% { transform: translateY(-10px) rotate(-5deg); }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: var(--bg-glass);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
        color: var(--text-secondary);
    }
    
    .footer a {
        color: var(--accent);
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: var(--accent-hover);
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .user-message, .bot-message {
            margin-left: 5% !important;
            margin-right: 5% !important;
        }
        
        .chat-container {
            padding: 1rem;
        }
        
        .chat-input-container, .stForm {
            padding: 1rem !important;
        }
        
        .stTextInput > div > div > input, 
        .stTextInput input,
        [data-testid="stTextInput"] > div > div > input,
        [data-testid="stTextInput"] input {
            font-size: 1rem !important;
            height: 3rem !important;
            padding: 0.75rem 1rem !important;
        }
        
        .stForm .stButton > button,
        .stForm button[kind="formSubmit"],
        button[kind="formSubmit"],
        [data-testid="baseButton-secondary"] {
            height: 3rem !important;
            font-size: 1rem !important;
        }
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-secondary);
    }
    
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent);
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) {
        animation-delay: 0.2s;
    }
    
    .typing-dot:nth-child(3) {
        animation-delay: 0.4s;
    }
    
    @keyframes typing {
        0%, 60%, 100% {
            transform: scale(1);
            opacity: 1;
        }
        30% {
            transform: scale(1.5);
            opacity: 0.7;
        }
    }
    
    /* Enhanced focus ring for accessibility */
    .stTextInput > div > div > input:focus-visible,
    .stTextInput input:focus-visible,
    [data-testid="stTextInput"] > div > div > input:focus-visible,
    [data-testid="stTextInput"] input:focus-visible {
        outline: 2px solid var(--accent) !important;
        outline-offset: 2px !important;
    }

    /* Additional button visibility fixes */
    .stForm button {
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-flex !important;
    }
    
    /* Force visibility on all form buttons */
    button[type="submit"],
    button[kind="formSubmit"] {
        opacity: 1 !important;
        visibility: visible !important;
        display: inline-flex !important;
        background: linear-gradient(135deg, var(--send-button) 0%, #16a34a 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border: 2px solid rgba(34, 197, 94, 0.8) !important;
        box-shadow: 0 6px 20px rgba(34, 197, 94, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    def __init__(self, api_key=None):
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client()
            
            # Create collection with unique name to avoid conflicts
            collection_name = f"documents_{int(time.time())}"
            try:
                # Try to get existing collection or create new one
                self.collection = self.client.get_or_create_collection(collection_name)
            except:
                self.collection = self.client.create_collection(collection_name)
            
            # Initialize sentence transformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize OpenRouter client (uses OpenAI-compatible API)
            self.openai_client = None
            if api_key and api_key.strip():
                try:
                    self.openai_client = openai.OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    # Test the API key with a simple request
                    test_response = self.openai_client.models.list()
                    self.api_key_valid = True
                except Exception as e:
                    self.api_key_valid = False
                    st.error(f"Invalid API key: {str(e)}")
            else:
                self.api_key_valid = False
            
            self.selected_model = 'meta-llama/llama-3.1-8b-instruct:free'
                
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            self.api_key_valid = False

    def load_pdf_from_upload(self, uploaded_file):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Read the PDF
            with open(tmp_file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            if not text.strip():
                return False, "No text could be extracted from the PDF"
            
            # Process text into chunks
            chunks = []
            sentences = text.split('. ')
            current_chunk = ""

            for sentence in sentences:
                if len(current_chunk + sentence) < 500:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + ". "

            if current_chunk:
                chunks.append(current_chunk.strip())

            if not chunks:
                return False, "No text chunks could be created from the PDF"

            # Clear existing collection and add new chunks
            try:
                # Delete existing items
                existing_ids = self.collection.get()['ids']
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except Exception as e:
                st.warning(f"Could not clear existing collection: {e}")

            # Add chunks with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.encoder.encode([chunk])[0].tolist()
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[chunk],
                        ids=[f"chunk_{i}"]
                    )
                    progress_bar.progress((i + 1) / len(chunks))
                    status_text.text(f"Processing chunk {i + 1} of {len(chunks)}")
                except Exception as e:
                    st.error(f"Error adding chunk {i}: {e}")
                    
            progress_bar.empty()
            status_text.empty()
            
            return True, len(chunks)
            
        except Exception as e:
            return False, str(e)

    def ask(self, query):
        try:
            if not self.openai_client or not self.api_key_valid:
                return "Error: OpenRouter API key not provided or invalid. Please add a valid API key in the sidebar."
            
            # Get embeddings for the query
            query_embedding = self.encoder.encode([query])[0].tolist()
            
            # Search for relevant chunks
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            if not results['documents'] or not results['documents'][0]:
                return "No relevant information found in the document. Please make sure you've uploaded and processed a document."
            
            context = "\n".join(results['documents'][0])

            # Use OpenRouter API (OpenAI-compatible)
            response = self.openai_client.chat.completions.create(
                model=self.selected_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. Answer questions based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."
                    },
                    {
                        "role": "user", 
                        "content": f"Context: {context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context above."
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
                
        except openai.AuthenticationError:
            return "Error: Invalid OpenRouter API key. Please check your API key."
        except openai.RateLimitError:
            return "Error: OpenRouter API rate limit exceeded. Please try again later."
        except openai.APIError as e:
            return f"Error: OpenRouter API error - {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'chunk_count' not in st.session_state:
        st.session_state.chunk_count = 0
    if 'current_api_key' not in st.session_state:
        st.session_state.current_api_key = ""
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🤖 RAG Chatbot AI</div>
        <div class="hero-subtitle">Transform your documents into intelligent conversations with cutting-edge AI</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # API Configuration Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🔑 API Configuration</div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "",
            type="password",
            help="Enter your OpenRouter API key. Get one from https://openrouter.ai/keys",
            placeholder="sk-or-v1-...",
            value=st.session_state.get('current_api_key', ''),
            label_visibility="collapsed"
        )
        
        # Initialize or reinitialize chatbot when API key changes
        if api_key != st.session_state.current_api_key or 'chatbot' not in st.session_state:
            st.session_state.current_api_key = api_key
            if api_key and api_key.strip():
                with st.spinner("🔍 Validating API key..."):
                    st.session_state.chatbot = RAGChatbot(api_key=api_key)
                if st.session_state.chatbot.api_key_valid:
                    st.markdown('<div class="status-card status-success">✅ API key validated successfully</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-card status-error">❌ Invalid API key provided</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-card status-warning">⚠️ Please enter your OpenRouter API key</div>', unsafe_allow_html=True)
                if 'chatbot' in st.session_state:
                    del st.session_state.chatbot
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Document Upload Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📁 Document Upload</div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "",
            type="pdf",
            help="Upload a PDF document to chat with its content",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            st.markdown(f"📄 **{uploaded_file.name}** ({uploaded_file.size} bytes)")
            
            if st.button("🚀 Process Document", use_container_width=True):
                if not api_key or not api_key.strip():
                    st.markdown('<div class="status-card status-error">❌ Please provide an OpenRouter API key first!</div>', unsafe_allow_html=True)
                elif 'chatbot' not in st.session_state or not st.session_state.chatbot.api_key_valid:
                    st.markdown('<div class="status-card status-error">❌ Please provide a valid OpenRouter API key first!</div>', unsafe_allow_html=True)
                else:
                    with st.spinner("🔄 Processing document..."):
                        success, result = st.session_state.chatbot.load_pdf_from_upload(uploaded_file)
                        
                    if success:
                        st.session_state.document_loaded = True
                        st.session_state.chunk_count = result
                        st.markdown(f"""
                        <div class="status-card status-success">
                            ✅ Document processed successfully!<br>
                            📊 Created {result} intelligent text chunks
                        </div>
                        """, unsafe_allow_html=True)
                        st.rerun()
                    else:
                        st.markdown(f"""
                        <div class="status-card status-error">
                            ❌ Processing failed: {result}
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Status Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📊 System Status</div>
        """, unsafe_allow_html=True)
        
        if not api_key or not api_key.strip():
            st.markdown('<div class="status-card status-error">🔑 API key required</div>', unsafe_allow_html=True)
        elif 'chatbot' not in st.session_state or not st.session_state.chatbot.api_key_valid:
            st.markdown('<div class="status-card status-error">🔑 Invalid API key</div>', unsafe_allow_html=True)
        elif st.session_state.document_loaded:
            st.markdown(f'<div class="status-card status-success">✅ Ready to chat ({st.session_state.chunk_count} chunks loaded)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card status-warning">⚠️ No document loaded</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model Settings Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🤖 AI Model Settings</div>
        """, unsafe_allow_html=True)
        
        model_option = st.selectbox(
            "",
            [
                "meta-llama/llama-3.1-8b-instruct:free",
                "meta-llama/llama-3.1-70b-instruct:free",
                "microsoft/phi-3-mini-128k-instruct:free",
                "google/gemma-2-9b-it:free",
                "openai/gpt-3.5-turbo",
                "openai/gpt-4o-mini",
                "openai/gpt-4o",
                "anthropic/claude-3-haiku",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro-1.5",
                "meta-llama/llama-3.1-405b-instruct"
            ],
            help="Free models available! Paid models offer better performance.",
            label_visibility="collapsed"
        )
        
        # Update model in chatbot
        if 'chatbot' in st.session_state:
            st.session_state.chatbot.selected_model = model_option
        
        # Show model info
        if "free" in model_option:
            st.markdown("💚 **Free Model** - No credits required")
        else:
            st.markdown("💎 **Premium Model** - Requires OpenRouter credits")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Actions Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🎛️ Actions</div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Main Content Area
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col2:
        # Chat Interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.chat_history:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: var(--text-secondary);">
                <h3 style="color: var(--text-primary); margin-bottom: 1rem;">👋 Welcome to RAG Chatbot AI</h3>
                <p>Upload a PDF document and start chatting with its content using advanced AI models.</p>
                <p>Your intelligent document companion is ready to answer questions!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"""
            <div class="user-message">
                <strong>You:</strong> {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="bot-message">
                <strong>AI Assistant:</strong> {bot_msg}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Input Section with enhanced visibility and improved layout
        st.markdown('<h3 class="input-section-title">💭 Ask your question</h3>', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            # Use a better column ratio for better button visibility
            col_input, col_button = st.columns([3.5, 1])
            
            with col_input:
                user_question = st.text_input(
                    "",
                    placeholder="What insights can you share about this document?",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col_button:
                # Use a more explicit button with better styling
                submit_button = st.form_submit_button(
                    "Send ✨",
                    use_container_width=True,
                    type="primary"
                )

        if submit_button and user_question:
            if not api_key or not api_key.strip():
                st.error("🔑 Please provide an OpenRouter API key in the sidebar!")
            elif 'chatbot' not in st.session_state or not st.session_state.chatbot.api_key_valid:
                st.error("🔑 Please provide a valid OpenRouter API key in the sidebar!")
            elif not st.session_state.document_loaded:
                st.error("📄 Please upload and process a document first!")
            else:
                # Show typing indicator
                typing_placeholder = st.empty()
                typing_placeholder.markdown("""
                <div class="typing-indicator">
                    <span>AI is thinking</span>
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                response = st.session_state.chatbot.ask(user_question)
                typing_placeholder.empty()
                    
                # Add to chat history
                st.session_state.chat_history.append((user_question, response))
                st.rerun()

    # Footer
    st.markdown("""
    <div class="footer">
        <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🚀 Powered by Advanced AI Technology</h4>
        <p>
            🔗 <a href="https://openrouter.ai/keys" target="_blank">Get your OpenRouter API key</a> |
            💚 Free models available including Llama 3.1, Phi-3, and Gemma-2
        </p>
        <p style="margin-top: 1rem;">
            Built with ❤️ using Streamlit, ChromaDB, Sentence Transformers & OpenRouter API
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span>🧠 Vector Search</span>
            <span>🤖 Multiple AI Models</span>
            <span>📊 Real-time Processing</span>
            <span>🎨 Modern UI/UX</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Floating background elements
    st.markdown("""
    <div class="floating-element" style="top: 10%; left: 10%; font-size: 2rem;">🤖</div>
    <div class="floating-element" style="top: 20%; right: 15%; font-size: 1.5rem;">✨</div>
    <div class="floating-element" style="bottom: 30%; left: 5%; font-size: 1.8rem;">📚</div>
    <div class="floating-element" style="bottom: 10%; right: 10%; font-size: 1.2rem;">🔮</div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
