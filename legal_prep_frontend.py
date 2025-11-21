# legal_prep_frontend.py
import streamlit as st
import requests
from datetime import datetime
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .analysis-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #1f4788;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        white-space: pre-wrap;
        font-family: 'Courier New', monospace;
        line-height: 1.6;
    }
    .for-text {
        color: #28a745;
        font-weight: bold;
    }
    .against-text {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Backend API URL
API_URL = "http://localhost:8000"

def check_api_health():
    """Check if backend API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def analyze_document(file):
    """Send document to backend for analysis"""
    files = {'file': (file.name, file, 'application/pdf')}
    
    try:
        with st.spinner('🔍 Analyzing document... This may take 2-3 minutes...'):
            response = requests.post(
                f"{API_URL}/api/analyze",
                files=files,
                timeout=300  # 5 minutes timeout
            )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Error: {response.status_code} - {response.text}"
    
    except requests.exceptions.Timeout:
        return None, "Request timed out. The document might be too large."
    except Exception as e:
        return None, f"Error connecting to API: {str(e)}"

def display_analysis_text(analysis_text):
    """Display the analysis text with formatting"""
    # Color code FOR and AGAINST categories
    formatted_text = analysis_text
    formatted_text = formatted_text.replace("Category: FOR", "<span class='for-text'>Category: FOR</span>")
    formatted_text = formatted_text.replace("Category: AGAINST", "<span class='against-text'>Category: AGAINST</span>")
    
    st.markdown(f"""
    <div class="analysis-box">
        {formatted_text}
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Strategic Legal Preparation Tool</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; color: #6c757d;'>
    AI-Powered Analysis for Comprehensive Case Preparation
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/law.png", width=80)
        st.title("About")
        st.info("""
        This tool uses advanced AI to analyze legal documents and extract:
        
        ✓ Top 10 pivotal case items
        ✓ Arguments FOR and AGAINST
        ✓ Precise page & line citations
        ✓ Strategic legal significance
        
        **Powered by:**
        - Google Gemini AI
        - LangChain
        - FAISS Vector Search
        """)
        
        st.markdown("---")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("✅ Backend API: Connected")
        else:
            st.error("❌ Backend API: Not Connected")
            st.warning("Please start the FastAPI backend:\n```bash\npython legal_prep_backend.py\n```")
        
        st.markdown("---")
        st.caption("© 2025 Legal Tech Solutions")
    
    # Main content
    if not check_api_health():
        st.error("⚠️ Backend API is not running. Please start the FastAPI server first.")
        st.code("python legal_prep_backend.py", language="bash")
        return
    
    # File upload section
    st.markdown('<h2 class="sub-header">📄 Upload Legal Document</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file (Amicus Brief, Legal Document, etc.)",
        type=['pdf'],
        help="Upload the legal document you want to analyze"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Analyze Document", use_container_width=True, type="primary"):
                # Analyze document
                result, error = analyze_document(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                    return
                
                # Store result in session state
                st.session_state['analysis_result'] = result
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Document", result['document_name'])
        with col2:
            st.metric("Total Pages", result['total_pages'])
        with col3:
            st.metric("Analysis Date", result['analysis_date'][:10])
        
        st.markdown("---")
        
        # Display analysis text
        st.markdown('<h2 class="sub-header">🎯 Top 10 Pivotal Items Analysis</h2>', unsafe_allow_html=True)
        
        display_analysis_text(result['analysis_result'])
        
        # Export options
        st.markdown('<h2 class="sub-header">📥 Export Results</h2>', unsafe_allow_html=True)
        
        # Export as Text
        text_content = f"""
LEGAL DOCUMENT ANALYSIS REPORT
==============================

Document: {result['document_name']}
Analysis Date: {result['analysis_date'][:10]}
Total Pages: {result['total_pages']}

{result['analysis_result']}
"""
        
        st.download_button(
            label="📄 Download as Text File",
            data=text_content,
            file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

if __name__ == "__main__":
    main()