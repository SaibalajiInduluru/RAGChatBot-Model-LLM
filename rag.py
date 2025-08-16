import streamlit as st
import PyPDF2
import hnswlib
import numpy as np
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

# Modern CSS with bright theme and enhanced sidebar visibility
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for bright theme */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-glass: rgba(255, 255, 255, 0.9);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --accent: #3b82f6;
        --accent-hover: #2563eb;
        --success: #10b981;
        --error: #ef4444;
        --warning: #f59e0b;
        --send-button: #10b981;
        --send-button-hover: #059669;
        --border-color: #e2e8f0;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Force Streamlit to use our theme */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%) !important;
    }
    
    /* Global styles */
    .main, .block-container {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        padding-top: 1rem !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Force sidebar visibility and styling */
    .css-1d391kg, 
    .stSidebar .css-1d391kg,
    .stSidebar,
    section[data-testid="stSidebar"],
    .css-1cypcdb,
    .css-17eq0hr {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 2px solid var(--border-color) !important;
        box-shadow: 2px 0 10px var(--shadow) !important;
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
        width: 300px !important;
        min-width: 300px !important;
    }
    
    /* Sidebar content visibility */
    .stSidebar > div,
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        padding: 1rem !important;
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
    }
    
    /* Sidebar elements */
    .stSidebar .stMarkdown,
    .stSidebar .stTextInput,
    .stSidebar .stFileUploader,
    .stSidebar .stSelectbox,
    .stSidebar .stButton,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stFileUploader,
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stButton {
        color: var(--text-primary) !important;
        visibility: visible !important;
        display: block !important;
        opacity: 1 !important;
    }
    
    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 2px solid rgba(59, 130, 246, 0.2);
        box-shadow: 0 10px 30px var(--shadow);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
    }
    
    /* Sidebar section styling */
    .sidebar-section {
        background: rgba(59, 130, 246, 0.05) !important;
        border: 2px solid rgba(59, 130, 246, 0.2) !important;
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px var(--shadow);
    }
    
    .sidebar-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        border-color: var(--accent) !important;
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Chat container */
    .chat-container {
        background: var(--bg-glass);
        border: 2px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 10px 30px var(--shadow);
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--accent), #8b5cf6);
        border-radius: 10px;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 8px 25px;
        margin: 1rem 0;
        margin-left: 15%;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        animation: slideInRight 0.3s ease-out;
    }
    
    .bot-message {
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 25px 25px 25px 8px;
        margin: 1rem 0;
        margin-right: 15%;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
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
    
    /* Enhanced button styling with MAXIMUM visibility for sidebar */
    .stButton > button,
    .stSidebar .stButton > button,
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: 3px solid #10b981 !important;
        border-radius: 15px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        width: 100% !important;
        height: 3.5rem !important;
        min-height: 3.5rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
        letter-spacing: 0.5px !important;
        text-transform: uppercase !important;
        position: relative !important;
        z-index: 10 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    .stButton > button:hover,
    .stSidebar .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.7) !important;
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        border-color: #059669 !important;
    }
    
    .stButton > button:active,
    .stSidebar .stButton > button:active,
    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.6) !important;
    }
    
    /* Special styling for "Process Document" button */
    .stSidebar .stButton > button:contains("Process Document"),
    section[data-testid="stSidebar"] .stButton > button:contains("Process Document") {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
        border-color: #f59e0b !important;
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5) !important;
    }
    
    /* Special styling for "Clear Chat History" button */
    .stSidebar .stButton > button:contains("Clear"),
    section[data-testid="stSidebar"] .stButton > button:contains("Clear") {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border-color: #ef4444 !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5) !important;
    }
    
    /* Force all button containers to be visible */
    .stButton,
    .stSidebar .stButton,
    section[data-testid="stSidebar"] .stButton {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
    }
    
    /* Ensure button wrapper divs are visible */
    .stButton > div,
    .stSidebar .stButton > div,
    section[data-testid="stSidebar"] .stButton > div {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 100% !important;
    }
    
    /* Additional button visibility fixes */
    button[kind="primary"],
    button[kind="secondary"],
    .stButton button[kind="primary"],
    .stButton button[kind="secondary"],
    .stSidebar button[kind="primary"],
    .stSidebar button[kind="secondary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: 3px solid #10b981 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        visibility: visible !important;
        opacity: 1 !important;
        display: inline-flex !important;
        min-height: 3.5rem !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
    }
    
    /* Input fields with bright theme */
    .stTextInput > div > div > input, 
    .stTextInput input,
    [data-testid="stTextInput"] > div > div > input,
    [data-testid="stTextInput"] input {
        background: white !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 1rem 1.5rem !important;
        font-size: 1rem !important;
        height: 3.5rem !important;
        box-shadow: 0 2px 10px var(--shadow) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextInput input:focus,
    [data-testid="stTextInput"] > div > div > input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), 0 4px 15px var(--shadow) !important;
        outline: none !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextInput input::placeholder,
    [data-testid="stTextInput"] > div > div > input::placeholder,
    [data-testid="stTextInput"] input::placeholder {
        color: var(--text-secondary) !important;
    }
    
    /* Sidebar input fields */
    .stSidebar .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: white !important;
        border: 2px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .stSidebar .stTextInput > div > div > input:focus,
    section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div > select {
        background: white !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed rgba(59, 130, 246, 0.5) !important;
        border-radius: 15px !important;
        background: rgba(59, 130, 246, 0.05) !important;
        padding: 2rem !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader:hover {
        border-color: var(--accent) !important;
        background: rgba(59, 130, 246, 0.1) !important;
    }
    
    .stFileUploader > div {
        color: var(--text-primary) !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Form container */
    .stForm, .chat-input-container {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid var(--border-color) !important;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px var(--shadow);
    }
    
    /* Input section title */
    .input-section-title {
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent) 0%, #8b5cf6 50%, #a855f7 100%) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        border: 2px solid var(--border-color);
        margin-top: 3rem;
        color: var(--text-secondary);
        box-shadow: 0 4px 15px var(--shadow);
    }
    
    .footer a {
        color: var(--accent);
        text-decoration: none;
        transition: color 0.3s ease;
    }
    
    .footer a:hover {
        color: var(--accent-hover);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: var(--text-secondary);
        background: rgba(59, 130, 246, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(59, 130, 246, 0.2);
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
        
        .sidebar-section {
            padding: 1rem;
        }
    }
    
    /* Force sidebar visibility on mobile */
    @media (max-width: 768px) {
        .stSidebar,
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

class RAGChatbot:
    def __init__(self, api_key=None):
        try:
            # Initialize Hnswlib index
            self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
            self.index = None
            self.documents = []
            self.document_embeddings = []
            
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

            # Clear existing data
            self.documents = []
            self.document_embeddings = []

            # Create embeddings and add chunks with progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.encoder.encode([chunk])[0]
                    embeddings.append(embedding)
                    self.documents.append(chunk)
                    progress_bar.progress((i + 1) / len(chunks))
                    status_text.text(f"Processing chunk {i + 1} of {len(chunks)}")
                except Exception as e:
                    st.error(f"Error processing chunk {i}: {e}")
                    
            # Create Hnswlib index
            if embeddings:
                self.document_embeddings = np.array(embeddings)
                self.index = hnswlib.Index(space='cosine', dim=self.dimension)
                self.index.init_index(max_elements=len(embeddings), ef_construction=200, M=16)
                self.index.add_items(self.document_embeddings, list(range(len(embeddings))))
                self.index.set_ef(50)
                    
            progress_bar.empty()
            status_text.empty()
            
            return True, len(chunks)
            
        except Exception as e:
            return False, str(e)

    def ask(self, query):
        try:
            if not self.openai_client or not self.api_key_valid:
                return "Error: OpenRouter API key not provided or invalid. Please add a valid API key in the sidebar."
            
            if not self.index or len(self.documents) == 0:
                return "No relevant information found in the document. Please make sure you've uploaded and processed a document."
            
            # Get embeddings for the query
            query_embedding = self.encoder.encode([query])[0]
            
            # Search for relevant chunks
            labels, distances = self.index.knn_query([query_embedding], k=min(3, len(self.documents)))
            
            # Get the most relevant documents
            relevant_docs = []
            for label in labels[0]:
                if label < len(self.documents):
                    relevant_docs.append(self.documents[label])
            
            if not relevant_docs:
                return "No relevant information found in the document."
            
            context = "\n".join(relevant_docs)

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

    # Sidebar with enhanced visibility
    with st.sidebar:
        st.markdown("# 🚀 RAG Chatbot Control Panel")
        
        # API Configuration Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🔑 API Configuration</div>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key. Get one from https://openrouter.ai/keys",
            placeholder="sk-or-v1-...",
            value=st.session_state.get('current_api_key', '')
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
        
        # Document Upload Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📁 Document Upload</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with its content"
        )
        
        if uploaded_file is not None:
            st.success(f"📄 **{uploaded_file.name}** ({uploaded_file.size} bytes)")
            
        if st.button("🚀 Process Document", use_container_width=True, type="primary", key="process_btn"):
            if not api_key or not api_key.strip():
                st.error("❌ Please provide an OpenRouter API key first!")
            elif 'chatbot' not in st.session_state or not st.session_state.chatbot.api_key_valid:
                st.error("❌ Please provide a valid OpenRouter API key first!")
            else:
                with st.spinner("🔄 Processing document..."):
                    success, result = st.session_state.chatbot.load_pdf_from_upload(uploaded_file)
                    
                if success:
                    st.session_state.document_loaded = True
                    st.session_state.chunk_count = result
                    st.success(f"✅ Document processed successfully! Created {result} intelligent text chunks")
                    st.rerun()
                else:
                    st.error(f"❌ Processing failed: {result}")

        # Status Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">📊 System Status</div>
        </div>
        """, unsafe_allow_html=True)
        
        if not api_key or not api_key.strip():
            st.warning("🔑 API key required")
        elif 'chatbot' not in st.session_state or not st.session_state.chatbot.api_key_valid:
            st.error("🔑 Invalid API key")
        elif st.session_state.document_loaded:
            st.success(f"✅ Ready to chat ({st.session_state.chunk_count} chunks loaded)")
        else:
            st.info("⚠️ No document loaded")
        
        # Model Settings Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🤖 AI Model Settings</div>
        </div>
        """, unsafe_allow_html=True)
        
        model_option = st.selectbox(
            "Select AI Model",
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
            help="Free models available! Paid models offer better performance."
        )
        
        # Update model in chatbot
        if 'chatbot' in st.session_state:
            st.session_state.chatbot.selected_model = model_option
        
        # Show model info
        if "free" in model_option:
            st.success("💚 **Free Model** - No credits required")
        else:
            st.info("💎 **Premium Model** - Requires OpenRouter credits")
        
        # Actions Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">🎛️ Actions</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Help Section
        st.markdown("""
        <div class="sidebar-section">
            <div class="sidebar-title">❓ Quick Help</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Getting Started:**
        1. 🔑 Enter your OpenRouter API key
        2. 📁 Upload a PDF document
        3. 🚀 Click "Process Document"
        4. 💬 Start chatting with your document!
        
        **Get API Key:** [openrouter.ai/keys](https://openrouter.ai/keys)
        """)

    # Main Content Area
    col1, col2, col3 = st.columns([1, 6, 1])
    
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

        # Input Section
        st.markdown('<h3 class="input-section-title">💭 Ask your question</h3>', unsafe_allow_html=True)
        
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])
            
            with col_input:
                user_question = st.text_input(
                    "Your question",
                    placeholder="What insights can you share about this document?",
                    label_visibility="collapsed",
                    key="user_input"
                )
            
            with col_button:
                submit_button = st.form_submit_button(
                    "SEND ✨",
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
            Built with ❤️ using Streamlit, Hnswlib, Sentence Transformers & OpenRouter API
        </p>
        <div style="margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <span>🧠 Vector Search</span>
            <span>🤖 Multiple AI Models</span>
            <span>📊 Real-time Processing</span>
            <span>🎨 Modern UI/UX</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
