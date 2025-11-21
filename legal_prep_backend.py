# legal_prep_backend.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import json
from datetime import datetime

app = FastAPI(title="Legal Document Analyzer API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response Models
class Citation(BaseModel):
    page: int
    line_range: str
    text_snippet: str

class KeyItem(BaseModel):
    rank: int
    title: str
    category: str  # "FOR" or "AGAINST"
    description: str
    legal_significance: str
    citations: List[Citation]

class AnalysisResponse(BaseModel):
    document_name: str
    analysis_date: str
    total_pages: int
    key_items: List[KeyItem]

# Configuration - Only Groq API key needed
GROQ_API_KEY = "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"

class DocumentProcessor:
    """Handles PDF processing with metadata extraction"""
    
    def __init__(self):
        self.chunks_with_metadata = []
        
    def extract_text_with_metadata(self, pdf_file) -> List[Dict]:
        """Extract text from PDF with page and line metadata"""
        reader = PdfReader(pdf_file)
        documents = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only process pages with text
                lines = text.split('\n')
                
                # Store each line with metadata
                for line_num, line in enumerate(lines, start=1):
                    if line.strip():  # Skip empty lines
                        documents.append({
                            'content': line.strip(),
                            'page': page_num,
                            'line': line_num,
                            'metadata': {
                                'page': page_num,
                                'line': line_num,
                                'source': pdf_file.name if hasattr(pdf_file, 'name') else 'document.pdf'
                            }
                        })
        
        return documents
    
    def create_chunks_with_metadata(self, documents: List[Dict]) -> List[Dict]:
        """Create larger chunks while preserving metadata - optimized for token limits"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = []
        current_chunk = ""
        current_metadata = []
        
        for doc in documents:
            if len(current_chunk) + len(doc['content']) > 1000:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'start_page': current_metadata[0]['page'],
                            'end_page': current_metadata[-1]['page'],
                            'start_line': current_metadata[0]['line'],
                            'end_line': current_metadata[-1]['line']
                        }
                    })
                current_chunk = doc['content']
                current_metadata = [doc['metadata']]
            else:
                if current_chunk:
                    current_chunk += " " + doc['content']
                else:
                    current_chunk = doc['content']
                current_metadata.append(doc['metadata'])
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'start_page': current_metadata[0]['page'],
                    'end_page': current_metadata[-1]['page'],
                    'start_line': current_metadata[0]['line'],
                    'end_line': current_metadata[-1]['line']
                }
            })
        
        return chunks

class VectorStoreManager:
    """Manages vector embeddings using local models"""
    
    def __init__(self):
        # Use local embeddings to avoid API quotas
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        
    def create_vector_store(self, chunks: List[Dict]):
        """Create FAISS vector store from chunks using local embeddings"""
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            for doc, score in results
        ]

class GroqAnalyzer:
    """Handles Groq API interactions for legal analysis"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.available_models = self._discover_available_models()
        
    def _discover_available_models(self):
        """Discover available models from Groq API"""
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models.data]
            print(f"Available Groq models: {available_models}")
            return available_models
        except Exception as e:
            print(f"Failed to discover models, using defaults: {e}")
            # Return commonly available models as fallback
            return [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "llama-3.1-405b-reasoning",
                "mixtral-8x7b-32768"
            ]
    
    def get_available_model(self):
        """Get the first available model from discovered models"""
        preferred_models = [
            "llama-3.1-70b-versatile",    # Best for complex analysis
            "llama-3.1-8b-instant",       # Fast and efficient
            "llama-3.1-405b-reasoning",   # Most capable
            "mixtral-8x7b-32768"          # Legacy
        ]
        
        for model in preferred_models:
            if model in self.available_models:
                return model
        
        # If no preferred models found, use first available
        if self.available_models:
            return self.available_models[0]
        else:
            raise HTTPException(status_code=500, detail="No available Groq models found")
        
    def extract_key_items(self, relevant_chunks: List[Dict]) -> str:
        """Extract top 10 key items using Groq API"""
        
        # Prepare context efficiently
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:12]):  # Limit to top 12 chunks
            context_parts.append(
                f"[Chunk {i+1}, Page {chunk['metadata']['start_page']}-{chunk['metadata']['end_page']}]: "
                f"{chunk['content'][:500]}"  # Limit chunk preview
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""As an expert legal analyst, review this Amicus Brief and identify the TOP 10 most pivotal legal arguments and points. Focus on extracting the most significant legal reasoning, evidence, and arguments from both sides.

DOCUMENT EXCERPTS:
{context}

ANALYSIS INSTRUCTIONS:
1. Identify exactly 10 key legal items
2. Categorize each as "FOR" (supporting plaintiffs) or "AGAINST" (opposing/defending)
3. Ensure balanced coverage of both perspectives
4. For each item, provide:
   - Clear, concise title
   - FOR/AGAINST category
   - 2-3 sentence description
   - Legal significance and strategic impact
   - Specific page citations from the document excerpts

CRITICAL: Use ONLY the page numbers and content from the provided excerpts. Do not invent or assume information.

OUTPUT FORMAT:

TOP 10 LEGAL ANALYSIS
=====================

1. [Argument Title]
   Category: FOR/AGAINST
   Description: [2-3 sentence explanation]
   Legal Significance: [Strategic importance]
   Citations: Page X [and Page Y if applicable]

2. [Argument Title]
   Category: FOR/AGAINST
   Description: [2-3 sentence explanation]
   Legal Significance: [Strategic importance]
   Citations: Page X [and Page Y if applicable]

... continue for all 10 items

BALANCE SUMMARY:
- FOR arguments: X items
- AGAINST arguments: Y items"""

        try:
            model_to_use = self.get_available_model()
            print(f"Using model: {model_to_use}")
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise legal analyst specializing in Amicus Briefs. Extract exact legal arguments with accurate page citations from the provided text only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model_to_use,
                temperature=0.1,  # Lower temperature for more consistent citations
                max_tokens=4096,
                top_p=0.9,
                stream=False,
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            # If the primary model fails, try any available model
            for model in self.available_models:
                try:
                    print(f"Trying fallback model: {model}")
                    chat_completion = self.client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a precise legal analyst specializing in Amicus Briefs. Extract exact legal arguments with accurate page citations from the provided text only."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model=model,
                        temperature=0.1,
                        max_tokens=4096,
                        top_p=0.9,
                        stream=False,
                    )
                    return chat_completion.choices[0].message.content
                except Exception:
                    continue
            
            raise HTTPException(
                status_code=500,
                detail=f"Legal analysis failed with all available models. Available models: {self.available_models}"
            )

# API Endpoints
@app.post("/api/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Main endpoint to analyze legal document and extract key items
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Step 1: Extract text with metadata
        processor = DocumentProcessor()
        documents = processor.extract_text_with_metadata(open(tmp_file_path, 'rb'))
        
        if not documents:
            os.unlink(tmp_file_path)
            raise HTTPException(status_code=400, detail="No text could be extracted from the PDF")
        
        chunks = processor.create_chunks_with_metadata(documents)
        
        # Step 2: Create vector store with local embeddings
        vector_manager = VectorStoreManager()
        vector_manager.create_vector_store(chunks)
        
        # Step 3: Semantic search for relevant legal content
        search_queries = [
            "legal argument claim reasoning",
            "evidence fact support",
            "opposition counterargument defense", 
            "statute regulation law violation",
            "public interest equity fairness",
            "precedent case law ruling",
            "constitutional right amendment"
        ]
        
        relevant_chunks = []
        for query in search_queries:
            try:
                results = vector_manager.semantic_search(query, k=2)
                relevant_chunks.extend(results)
            except Exception as e:
                print(f"Search query failed: {query}, error: {e}")
                continue
        
        # Remove duplicates based on content
        unique_chunks = []
        seen_content = set()
        for chunk in relevant_chunks:
            content_hash = hash(chunk['content'][:100])  # Hash first 100 chars for deduplication
            if content_hash not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content_hash)
        
        # Step 4: Analyze with Groq
        analyzer = GroqAnalyzer(GROQ_API_KEY)
        analysis_text = analyzer.extract_key_items(unique_chunks)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        # Prepare response
        return {
            "document_name": file.filename,
            "analysis_date": datetime.now().isoformat(),
            "total_pages": len(set(doc['page'] for doc in documents)),
            "analysis_result": analysis_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        models = client.models.list()
        available_models = [model.id for model in models.data]
        
        return {
            "status": "healthy",
            "groq_configured": bool(GROQ_API_KEY),
            "embeddings": "local-sentence-transformers",
            "available_models": available_models
        }
    except Exception as e:
        return {
            "status": "degraded",
            "groq_configured": bool(GROQ_API_KEY),
            "embeddings": "local-sentence-transformers",
            "error": str(e)
        }

@app.get("/available-models")
async def get_available_models():
    """Endpoint to check available Groq models"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        models = client.models.list()
        available_models = [model.id for model in models.data]
        return {
            "available_models": available_models,
            "count": len(available_models)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)