import streamlit as st
import requests
import time

# Backend API URL
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .similarity-score {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'document_id' not in st.session_state:
    st.session_state.document_id = None
if 'document_name' not in st.session_state:
    st.session_state.document_name = None


def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_documents(uploaded_files):
    """Upload multiple documents to backend"""
    try:
        files = [
            ("files", (file.name, file, file.type))
            for file in uploaded_files
        ]
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error uploading documents: {str(e)}")
        return None


def ask_question(question, document_id=None, top_k=3):
    """Send question to backend"""
    try:
        payload = {
            "question": question,
            "document_id": document_id,
            "top_k": top_k
        }
        response = requests.post(f"{API_URL}/chat", json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error asking question: {str(e)}")
        return None


# Main UI
st.markdown('<div class="main-header"> Simple RAG System</div>', unsafe_allow_html=True)

# Check backend health
if not check_backend_health():
    st.error(" Backend is not running! Please start the FastAPI server.")
    st.info("Run: `python main.py` in the backend directory")
    st.stop()

st.success(" Backend is connected")

# Sidebar for document upload
with st.sidebar:
    st.header(" Document Upload")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=['pdf', 'txt', 'md'],
        help="Upload PDF, TXT, or Markdown files",
        accept_multiple_files=True  # Enable multiple files
    )
    
    if uploaded_files:
        st.info(f"Selected {len(uploaded_files)} file(s)")
        for file in uploaded_files:
            st.write(f" {file.name}")
        
        if st.button(" Process All Documents", type="primary", use_container_width=True):
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
                result = upload_documents(uploaded_files)
                
                if result:
                    st.success(f" {result['message']}")
                    
                    # Display results for each document
                    st.subheader("Processing Results:")
                    for doc in result['documents']:
                        if doc['status'] == 'success':
                            st.success(f"✓ {doc['filename']}")
                            st.json({
                                "Document ID": doc['document_id'],
                                "Chunks": doc['chunks_count'],
                                "Size": f"{doc['file_size'] / 1024:.1f} KB"
                            })
                        else:
                            st.error(f"✗ {doc['filename']}: {doc.get('error', 'Unknown error')}")
                    
                    st.info(f"**Total:** {result['total_documents']} documents, {result['total_chunks']} chunks")
                    
                    # Clear old state and refresh
                    st.session_state.messages = []
                    time.sleep(1)
                    st.rerun()
    
    st.divider()
    
    # Document list
    st.subheader(" Uploaded Documents")
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json()['documents']
            
            if docs:
                for doc in docs:
                    with st.expander(f" {doc['filename']}"):
                        st.write(f"**ID:** {doc['id']}")
                        st.write(f"**Type:** {doc['file_type'].upper()}")
                        st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                        st.write(f"**Uploaded:** {doc['uploaded_at'][:19]}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(" Download", key=f"download_{doc['id']}", use_container_width=True):
                                download_url = f"{API_URL}/document/{doc['id']}/download"
                                st.markdown(f"[Click to download]({download_url})")
                        with col2:
                            if st.button(" Delete", key=f"delete_{doc['id']}", use_container_width=True):
                                requests.delete(f"{API_URL}/document/{doc['id']}")
                                st.success("Deleted!")
                                time.sleep(0.5)
                                st.rerun()
            else:
                st.info("No documents uploaded yet")
    except Exception as e:
        st.error(f"Error loading documents: {e}")
    
    st.divider()
    
    # Settings
    st.subheader(" Settings")
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    
    # Document selection for chat
    st.subheader(" Chat Scope")
    chat_scope = st.radio(
        "Search in:",
        ["All Documents", "Select Specific Document"],
        help="Choose whether to search across all documents or a specific one"
    )
    
    selected_doc_id = None
    if chat_scope == "Select Specific Document" and docs:
        doc_options = {f"{doc['filename']} (ID: {doc['id']})": doc['id'] for doc in docs}
        selected = st.selectbox("Choose document:", list(doc_options.keys()))
        selected_doc_id = doc_options[selected]
    
    # Store in session state
    st.session_state.selected_doc_id = selected_doc_id

# Main chat interface
st.header(" Chat with Your Document")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(" View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-box">
                        <b>Source {i+1}</b> (Similarity: <span class="similarity-score">{source['similarity']:.2%}</span>)<br>
                        <small><i>From: {source['filename']}</i></small><br><br>
                        {source['text']}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if any documents exist
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs = response.json()['documents']
            if not docs:
                st.warning(" Please upload documents first using the sidebar")
                st.stop()
    except:
        pass
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Use selected document or search all
            doc_id = st.session_state.get('selected_doc_id', None)
            
            response = ask_question(
                prompt, 
                doc_id,
                top_k
            )
            
            if response:
                st.markdown(response['answer'])
                
                # Store assistant message with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response['sources']
                })
                
                # Display sources
                with st.expander("📎 View Sources"):
                    for i, source in enumerate(response['sources']):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Source {i+1}</b> (Similarity: <span class="similarity-score">{source['similarity']:.2%}</span>)<br>
                            <small><i>From: {source['filename']}</i></small><br><br>
                            {source['text']}
                        </div>
                        """, unsafe_allow_html=True)
                
                st.rerun()

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    Built with  using FastAPI, LangChain, PostgreSQL (pgvector), and Streamlit
</div>
""", unsafe_allow_html=True)