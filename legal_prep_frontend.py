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

# Custom CSS with improved styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    
    /* Improved Card Styles */
    .for-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e1f5fe 100%);
        border: 1px solid #4fc3f7;
        border-left: 6px solid #0288d1;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(2, 136, 209, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .for-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(2, 136, 209, 0.15);
    }
    
    .against-card {
        background: linear-gradient(135deg, #fff0f0 0%, #ffeaea 100%);
        border: 1px solid #ff8a8a;
        border-left: 6px solid #d32f2f;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(211, 47, 47, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .against-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(211, 47, 47, 0.15);
    }
    
    .item-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #1a237e;
        line-height: 1.4;
    }
    
    .category-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .for-badge {
        background-color: #0288d1;
        color: white;
    }
    .against-badge {
        background-color: #d32f2f;
        color: white;
    }
    
    .description-text {
        color: #37474f;
        line-height: 1.6;
        margin-bottom: 1rem;
        font-size: 1rem;
    }
    
    .significance-section {
        background: rgba(255, 255, 255, 0.7);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #ffa000;
    }
    .significance-title {
        font-weight: 600;
        color: #ff6f00;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .significance-text {
        color: #5d4037;
        line-height: 1.5;
        font-style: italic;
    }
    
    .citations-section {
        margin-top: 1.2rem;
    }
    .citation-title {
        font-weight: 600;
        color: #455a64;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .citation-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        border: 1px solid #e0e0e0;
        transition: background-color 0.2s ease;
    }
    .citation-item:hover {
        background: rgba(255, 255, 255, 1);
    }
    .citation-location {
        color: #0288d1;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .citation-text {
        color: #546e7a;
        font-size: 0.9rem;
        line-height: 1.4;
        font-style: italic;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    
    /* Statistics Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 4px solid;
    }
    .stat-for {
        border-top-color: #0288d1;
    }
    .stat-against {
        border-top-color: #d32f2f;
    }
    .stat-total {
        border-top-color: #7b1fa2;
    }
    .stat-time {
        border-top-color: #388e3c;
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

def display_structured_analysis(result):
    """Display the structured analysis with key items"""
    
    # Summary statistics
    for_count = sum(1 for item in result['key_items'] if item['category'] == 'FOR')
    against_count = sum(1 for item in result['key_items'] if item['category'] == 'AGAINST')
    
    st.markdown("### 📈 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card stat-for">
            <div style="font-size: 2rem; color: #0288d1; margin-bottom: 0.5rem;">✅</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #0288d1;">{for_count}</div>
            <div style="color: #546e7a; font-size: 0.9rem;">FOR Arguments</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card stat-against">
            <div style="font-size: 2rem; color: #d32f2f; margin-bottom: 0.5rem;">❌</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #d32f2f;">{against_count}</div>
            <div style="color: #546e7a; font-size: 0.9rem;">AGAINST Arguments</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card stat-total">
            <div style="font-size: 2rem; color: #7b1fa2; margin-bottom: 0.5rem;">📊</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #7b1fa2;">{len(result['key_items'])}</div>
            <div style="color: #546e7a; font-size: 0.9rem;">Total Items</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        if result.get('processing_time'):
            st.markdown(f"""
            <div class="stat-card stat-time">
                <div style="font-size: 2rem; color: #388e3c; margin-bottom: 0.5rem;">⏱️</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #388e3c;">{result['processing_time']:.1f}s</div>
                <div style="color: #546e7a; font-size: 0.9rem;">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display each key item
    st.markdown("### 🎯 Key Legal Arguments")
    
    for item in result['key_items']:
        category_class = "for-card" if item['category'] == 'FOR' else "against-card"
        category_badge_class = "for-badge" if item['category'] == 'FOR' else "against-badge"
        
        st.markdown(f"""
        <div class="{category_class}">
            <div class="rank-badge">#{item['rank']}</div>
            <div class="item-title">{item['title']}</div>
            <div class="category-badge {category_badge_class}">{item['category']}</div>
            <div class="description-text">{item['description']}</div>
            
            <div class="significance-section">
                <div class="significance-title">Legal Significance</div>
                <div class="significance-text">{item['legal_significance']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display citations
        if item['citations']:
            st.markdown("""
            <div class="citations-section">
                <div class="citation-title">📍 Citations & Evidence</div>
            """, unsafe_allow_html=True)
            
            for citation in item['citations']:
                st.markdown(f"""
                <div class="citation-item">
                    <div class="citation-location">
                        📄 Page {citation['page']} • 📏 Lines {citation['line_range']}
                    </div>
                    <div class="citation-text">"{citation['text_snippet']}"</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_raw_analysis(raw_analysis):
    """Display raw analysis text as fallback"""
    if raw_analysis:
        # Try to pretty print JSON if it's JSON
        try:
            parsed_json = json.loads(raw_analysis)
            formatted_json = json.dumps(parsed_json, indent=2)
            st.markdown("### 📋 Raw Analysis Data")
            st.code(formatted_json, language='json')
        except:
            # If not JSON, display as text
            st.markdown("### 📋 Raw Analysis")
            st.text_area("", raw_analysis, height=400)

def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; font-size: 1.1rem; color: #6c757d; margin-bottom: 3rem;'>
    AI-Powered Analysis for Strategic Legal Preparation • Extract Key Arguments & Citations
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>⚖️</div>
            <h2 style='color: #1f4788; margin: 0;'>Legal Analyzer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### About")
        st.info("""
        **Extract from legal documents:**
        
        • ✅ FOR arguments
        • ❌ AGAINST arguments  
        • 📄 Page citations
        • 🎯 Legal significance
        • 📝 Text evidence
        
        **Powered by:**
        - Groq AI
        - LangChain
        - FAISS Vector Search
        """)
        
        st.markdown("---")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("**✅ API Status:** Connected")
        else:
            st.error("**❌ API Status:** Disconnected")
            st.warning("Start backend server:\n```bash\npython legal_prep_backend.py\n```")
        
        st.markdown("---")
        st.caption("© 2025 Legal Tech Analytics")

    # Main content
    if not check_api_health():
        st.error("🚨 **Backend API is not running**")
        st.markdown("""
        Please start the FastAPI backend server first:
        
        ```bash
        python legal_prep_backend.py
        ```
        
        The backend should run on: **http://localhost:8000**
        """)
        return
    
    # File upload section
    st.markdown('<h2 class="sub-header">📤 Upload Legal Document</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file to analyze (Amicus Brief, Legal Document, etc.)",
        type=['pdf'],
        help="Upload the legal document you want to analyze for key arguments and citations"
    )
    
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.success(f"**📄 Document Uploaded:** {uploaded_file.name}")
            
            if st.button("🚀 **Analyze Document**", use_container_width=True, type="primary"):
                # Analyze document
                result, error = analyze_document(uploaded_file)
                
                if error:
                    st.error(f"❌ **Analysis Failed:** {error}")
                    return
                
                # Store result in session state
                st.session_state['analysis_result'] = result
                st.success("✅ **Analysis Complete!** Scroll down to view results.")
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']
        
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
        
        # Document info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 Document", result['document_name'])
        with col2:
            st.metric("📄 Total Pages", result['total_pages'])
        with col3:
            st.metric("📅 Analysis Date", result['analysis_date'][:10])
        
        # Display structured analysis
        if 'key_items' in result and result['key_items']:
            display_structured_analysis(result)
        else:
            st.warning("⚠️ No structured analysis available. Showing raw output.")
            if 'raw_analysis' in result:
                display_raw_analysis(result['raw_analysis'])
        
        # Export options
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📥 Export Results</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as Text
            if 'key_items' in result:
                text_content = generate_text_report(result)
                st.download_button(
                    label="📄 Download Text Report",
                    data=text_content,
                    file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with col2:
            # Export as JSON
            if 'key_items' in result:
                json_content = json.dumps(result, indent=2)
                st.download_button(
                    label="📊 Download JSON Data",
                    data=json_content,
                    file_name=f"legal_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

def generate_text_report(result):
    """Generate a formatted text report from the analysis results"""
    report = []
    report.append("⚖️ LEGAL DOCUMENT ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Document: {result['document_name']}")
    report.append(f"Analysis Date: {result['analysis_date'][:10]}")
    report.append(f"Total Pages: {result['total_pages']}")
    if result.get('processing_time'):
        report.append(f"Processing Time: {result['processing_time']:.1f}s")
    report.append("")
    
    if 'key_items' in result:
        for_count = sum(1 for item in result['key_items'] if item['category'] == 'FOR')
        against_count = sum(1 for item in result['key_items'] if item['category'] == 'AGAINST')
        
        report.append("SUMMARY STATISTICS")
        report.append("-" * 20)
        report.append(f"FOR Arguments: {for_count}")
        report.append(f"AGAINST Arguments: {against_count}")
        report.append(f"Total Items: {len(result['key_items'])}")
        report.append("")
        
        report.append("KEY LEGAL ARGUMENTS")
        report.append("=" * 50)
        
        for item in result['key_items']:
            report.append(f"\n{item['rank']}. {item['title']}")
            report.append(f"   Category: {item['category']}")
            report.append(f"   Description: {item['description']}")
            report.append(f"   Legal Significance: {item['legal_significance']}")
            
            if item['citations']:
                report.append("   Citations:")
                for citation in item['citations']:
                    report.append(f"     - Page {citation['page']}, Lines {citation['line_range']}")
                    report.append(f"       \"{citation['text_snippet']}\"")
    
    return "\n".join(report)

if __name__ == "__main__":
    main()