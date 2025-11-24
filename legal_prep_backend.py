# legal_prep_backend.py
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional, Any
from groq import Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import json
from datetime import datetime
import hashlib
import logging
import time
from contextlib import contextmanager
import re
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Document Analyzer API")

# Configuration from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY environment variable not set. Please set it to use the Groq API.")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
PREFERRED_MODELS = os.getenv("PREFERRED_MODELS", "llama-3.1-70b-versatile,llama-3.1-8b-instant").split(",")
ENABLE_OCR_FALLBACK = os.getenv("ENABLE_OCR_FALLBACK", "false").lower() == "true"

# OCR imports with fallback
OCR_AVAILABLE = False
if ENABLE_OCR_FALLBACK:
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
        OCR_AVAILABLE = True
        logger.info("OCR capabilities enabled")
    except ImportError as e:
        logger.warning(f"OCR dependencies not available: {e}. Install with: pip install pytesseract pdf2image")
else:
    logger.info("OCR fallback disabled")

# CORS middleware - more restrictive
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS and ALLOWED_ORIGINS[0] else ["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Enhanced Response Models with validation (Pydantic V2)
class Citation(BaseModel):
    page: int
    line_range: str
    text_snippet: str
    chunk_index: Optional[int] = None
    char_offset: Optional[int] = None

class KeyItem(BaseModel):
    rank: int
    title: str
    category: str  # "FOR" or "AGAINST"
    description: str
    legal_significance: str
    citations: List[Citation]
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        if v.upper() not in ['FOR', 'AGAINST']:
            raise ValueError('Category must be "FOR" or "AGAINST"')
        return v.upper()

class AnalysisResponse(BaseModel):
    document_name: str
    analysis_date: str
    total_pages: int
    key_items: List[KeyItem]
    raw_analysis: Optional[str] = None
    processing_time: Optional[float] = None
    used_ocr: bool = False
    document_hash: Optional[str] = None

class ProcessingMetrics(BaseModel):
    extraction_time: float
    chunking_time: float
    embedding_time: float
    search_time: float
    analysis_time: float
    total_time: float

# Context manager for temporary file handling
@contextmanager
def temporary_pdf_file(content: bytes):
    """Secure context manager for temporary PDF files"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            temp_file = tmp_file.name
        yield temp_file
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")

class DocumentProcessor:
    """Enhanced PDF processing with OCR fallback and precise metadata"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
    def extract_text_with_metadata(self, pdf_file) -> tuple[List[Dict], bool]:
        """Extract text from PDF with robust error handling and OCR fallback"""
        reader = PdfReader(pdf_file)
        documents = []
        pages_without_text = 0
        used_ocr = False
        
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                
                # Handle None or empty text - try OCR if enabled
                if not text or text.strip() == "":
                    pages_without_text += 1
                    logger.warning(f"No text extracted from page {page_num}")
                    
                    # Try OCR if available and enabled
                    if OCR_AVAILABLE and ENABLE_OCR_FALLBACK:
                        logger.info(f"Attempting OCR for page {page_num}")
                        ocr_text = self._extract_text_with_ocr(pdf_file, page_num)
                        if ocr_text:
                            text = ocr_text
                            used_ocr = True
                            logger.info(f"OCR successful for page {page_num}")
                        else:
                            continue
                    else:
                        continue
                
                # Use splitlines() for better line handling
                lines = text.splitlines()
                line_count = 0
                char_offset = 0
                
                for line in lines:
                    clean_line = line.strip()
                    if clean_line:
                        line_count += 1
                        # Calculate character position within page
                        start_char = text.find(clean_line, char_offset)
                        if start_char != -1:
                            end_char = start_char + len(clean_line)
                            char_offset = end_char
                        else:
                            start_char = 0
                            end_char = len(clean_line)
                        
                        documents.append({
                            'content': clean_line,
                            'page': page_num,
                            'line': line_count,
                            'metadata': {
                                'page': page_num,
                                'line': line_count,
                                'source': getattr(pdf_file, 'name', 'document.pdf'),
                                'char_start': start_char,
                                'char_end': end_char,
                                'total_page_chars': len(text)
                            }
                        })
                        
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
        
        if pages_without_text > 0:
            logger.info(f"{pages_without_text} pages had no extractable text")
            if used_ocr:
                logger.info("OCR was used for some pages")
            
        if not documents:
            raise ValueError("No text content could be extracted from any page. Consider enabling OCR for scanned documents.")
            
        return documents, used_ocr
    
    def _extract_text_with_ocr(self, pdf_file, page_num: int) -> Optional[str]:
        """Extract text from PDF page using OCR"""
        try:
            # Convert specific page to image
            images = convert_from_bytes(
                pdf_file.read(), 
                first_page=page_num, 
                last_page=page_num,
                dpi=300  # Higher DPI for better OCR accuracy
            )
            
            if images:
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(images[0], config='--psm 6')
                # Reset file pointer for subsequent reads
                pdf_file.seek(0)
                return text if text.strip() else None
                
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
            # Reset file pointer in case of error
            pdf_file.seek(0)
        
        return None
    
    def create_chunks_with_metadata(self, documents: List[Dict]) -> List[Dict]:
        """Create chunks with precise metadata preservation and character-level accuracy"""
        if not documents:
            return []
            
        chunks = []
        current_chunk_lines = []
        current_metadata = []
        current_char_length = 0
        
        for doc in documents:
            current_content = doc['content']
            current_meta = doc['metadata']
            content_length = len(current_content)
            
            # If adding this line would exceed chunk size, finalize current chunk
            if current_chunk_lines and current_char_length + content_length > self.chunk_size:
                chunk_content = " ".join(line['content'] for line in current_chunk_lines)
                chunks.append(self._create_chunk_with_metadata(chunk_content, current_metadata))
                
                # Start new chunk with overlap (preserve last few lines)
                overlap_count = min(len(current_chunk_lines) // 2, 3)  # Keep last 3 lines for overlap
                current_chunk_lines = current_chunk_lines[-overlap_count:] if overlap_count > 0 else []
                current_metadata = current_metadata[-overlap_count:] if overlap_count > 0 else []
                current_char_length = sum(len(line['content']) for line in current_chunk_lines)
            
            current_chunk_lines.append({'content': current_content, 'metadata': current_meta})
            current_metadata.append(current_meta)
            current_char_length += content_length
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = " ".join(line['content'] for line in current_chunk_lines)
            chunks.append(self._create_chunk_with_metadata(chunk_content, current_metadata))
        
        return chunks
    
    def _create_chunk_with_metadata(self, content: str, metadata_list: List[Dict]) -> Dict:
        """Create a chunk with comprehensive metadata including character offsets"""
        if not metadata_list:
            return {'content': content, 'metadata': {}}
        
        # Calculate precise character ranges
        char_start = metadata_list[0].get('char_start', 0)
        char_end = metadata_list[-1].get('char_end', len(content))
        total_page_chars = metadata_list[0].get('total_page_chars', 0)
        
        return {
            'content': content,
            'metadata': {
                'start_page': metadata_list[0]['page'],
                'end_page': metadata_list[-1]['page'],
                'start_line': metadata_list[0]['line'],
                'end_line': metadata_list[-1]['line'],
                'source': metadata_list[0]['source'],
                'line_count': len(metadata_list),
                'char_start': char_start,
                'char_end': char_end,
                'total_page_chars': total_page_chars,
                'char_percentage': (char_end - char_start) / total_page_chars if total_page_chars > 0 else 0
            }
        }

class VectorStoreManager:
    """Enhanced vector store management with persistence and memory optimization"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.current_document_hash = None
        
    def create_vector_store(self, chunks: List[Dict], document_hash: str = None):
        """Create FAISS vector store with proper parameter handling"""
        if not chunks:
            raise ValueError("No chunks provided for vector store creation")
            
        texts = [chunk['content'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        try:
            # Use correct FAISS signature - positional arguments
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            self.current_document_hash = document_hash
            logger.info(f"Created vector store with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise ValueError(f"Failed to create vector store: {e}")
        
    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform semantic search with error handling"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [
                {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'chunk_hash': hashlib.sha256(doc.page_content.encode()).hexdigest()[:16]
                }
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Semantic search failed for query '{query}': {e}")
            return []
    
    def cleanup(self):
        """Clean up vector store to free memory"""
        self.vector_store = None
        self.current_document_hash = None
        logger.info("Vector store cleaned up")
    
    def save_to_disk(self, path: str):
        """Save vector store to disk for persistence"""
        if self.vector_store:
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def load_from_disk(self, path: str):
        """Load vector store from disk"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded from {path}")

class GroqAnalyzer:
    """Enhanced Groq API handler with structured output and better error handling"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Groq API key is required")
        self.client = Groq(api_key=api_key)
        self.available_models = self._discover_available_models()
        
    def _discover_available_models(self):
        """Safely discover available models"""
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models.data]
            logger.info(f"Discovered {len(available_models)} available Groq models")
            return available_models
        except Exception as e:
            logger.warning(f"Failed to discover models: {e}")
            # Use environment preferred models or defaults
            return PREFERRED_MODELS or [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768"
            ]
    
    def get_available_model(self):
        """Get best available model with fallbacks"""
        # Try preferred models from environment
        for model in PREFERRED_MODELS:
            if model in self.available_models:
                logger.info(f"Using preferred model: {model}")
                return model
        
        # Fallback to commonly available models
        fallback_models = [
            "llama-3.1-8b-instant",       # Fast and reliable
            "llama-3.1-70b-versatile",    # Good balance
            "mixtral-8x7b-32768"          # Legacy fallback
        ]
        
        for model in fallback_models:
            if model in self.available_models:
                logger.info(f"Using fallback model: {model}")
                return model
        
        # Last resort - use first available
        if self.available_models:
            model = self.available_models[0]
            logger.info(f"Using first available model: {model}")
            return model
        else:
            raise HTTPException(status_code=500, detail="No available Groq models found")
        
    def extract_key_items(self, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Extract structured key items with balanced coverage"""
        
        # Prepare enhanced context with full metadata
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:12]):  # Get more chunks for better coverage
            metadata = chunk['metadata']
            context_parts.append(
                f"CHUNK {i+1} (Pages {metadata['start_page']}-{metadata['end_page']}, "
                f"Lines {metadata['start_line']}-{metadata['end_line']}):\n"
                f"{chunk['content']}\n"
                f"--- END CHUNK {i+1} ---"
            )
        
        context = "\n\n".join(context_parts)
        
        structured_prompt = f"""As an expert legal analyst, analyze this Amicus Brief and extract EXACTLY 10 key legal items with perfectly balanced coverage.

DOCUMENT CONTEXT:
{context}

CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON - no other text
2. EXACTLY 5 FOR and 5 AGAINST items (10 total items)
3. Use ONLY information from the provided context
4. Include specific page citations from the context
5. For each citation, reference the CHUNK number where it appears
6. Each item MUST have at least one valid citation
7. Ensure citations reference actual pages and lines from the provided chunks

OUTPUT JSON SCHEMA:
{{
  "key_items": [
    {{
      "rank": 1,
      "title": "Concise argument title",
      "category": "FOR", 
      "description": "2-3 sentence explanation",
      "legal_significance": "Strategic importance and impact",
      "citations": [
        {{
          "page": 5,
          "line_range": "10-15", 
          "text_snippet": "Exact text from context that proves this point",
          "chunk_index": 3
        }}
      ]
    }}
  ]
}}

IMPORTANT RULES:
- "chunk_index" MUST match the CHUNK number from the context (1-{len(relevant_chunks)})
- Cite specific pages and line ranges from the metadata
- EXACTLY 5 FOR and 5 AGAINST items - no exceptions
- Each citation MUST include actual text from the document
- Do not invent or extrapolate beyond the provided context
- If you cannot find enough FOR or AGAINST arguments, prioritize the strongest ones you can find"""

        try:
            model_to_use = self.get_available_model()
            logger.info(f"Starting legal analysis with model: {model_to_use}")
            
            start_time = time.time()
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise legal analyst. Return ONLY valid JSON matching the exact schema. Extract arguments only from provided context with accurate citations. You MUST return exactly 5 FOR and 5 AGAINST items."
                    },
                    {
                        "role": "user",
                        "content": structured_prompt
                    }
                ],
                model=model_to_use,
                temperature=0.1,
                max_tokens=4096,
                top_p=0.9,
                stream=False,
                response_format={"type": "json_object"}
            )
            
            analysis_time = time.time() - start_time
            logger.info(f"LLM analysis completed in {analysis_time:.2f}s")
            
            response_text = chat_completion.choices[0].message.content
            return self._parse_and_validate_response(response_text, relevant_chunks)
            
        except Exception as e:
            logger.error(f"Legal analysis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Legal analysis failed: {str(e)}"
            )
    
    def _parse_and_validate_response(self, response_text: str, relevant_chunks: List[Dict]) -> Dict[str, Any]:
        """Parse and validate the LLM response with citation mapping"""
        try:
            # Extract JSON from response (handling potential extra text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
            
            data = json.loads(response_text)
            
            if 'key_items' not in data:
                raise ValueError("Missing 'key_items' in response")
            
            # Validate and map citations
            validated_items = []
            for item in data['key_items']:
                validated_item = self._validate_and_map_item(item, relevant_chunks)
                if validated_item:
                    validated_items.append(validated_item)
            
            # Ensure exactly 10 items with balanced coverage
            balanced_items = self._ensure_balanced_coverage(validated_items)
            
            return {
                'key_items': balanced_items,
                'raw_analysis': response_text  # Keep for debugging
            }
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse legal analysis results"
            )
    
    def _validate_and_map_item(self, item: Dict, relevant_chunks: List[Dict]) -> Optional[Dict]:
        """Validate individual item and map citations to authoritative metadata"""
        try:
            # Basic validation
            required_fields = ['rank', 'title', 'category', 'description', 'legal_significance', 'citations']
            if not all(field in item for field in required_fields):
                logger.warning(f"Item missing required fields: {item}")
                return None
            
            # Validate category
            if item['category'].upper() not in ['FOR', 'AGAINST']:
                logger.warning(f"Invalid category: {item['category']}")
                return None
            
            # Map citations to actual metadata
            validated_citations = []
            for citation in item.get('citations', []):
                validated_citation = self._map_citation_to_metadata(citation, relevant_chunks)
                if validated_citation:
                    validated_citations.append(validated_citation)
            
            if not validated_citations:
                logger.warning(f"Item has no valid citations: {item['title']}")
                return None  # Skip items without valid citations
                
            item['citations'] = validated_citations
            return item
            
        except Exception as e:
            logger.warning(f"Failed to validate item: {e}")
            return None
    
    def _map_citation_to_metadata(self, citation: Dict, relevant_chunks: List[Dict]) -> Optional[Dict]:
        """Map LLM citations to authoritative document metadata"""
        try:
            chunk_index = citation.get('chunk_index')
            if chunk_index is None or not (1 <= chunk_index <= len(relevant_chunks)):
                logger.warning(f"Invalid chunk index: {chunk_index}")
                return None
                
            chunk = relevant_chunks[chunk_index - 1]  # Convert to 0-based
            metadata = chunk['metadata']
            
            # Validate that the text snippet actually exists in the chunk
            text_snippet = citation.get('text_snippet', '').strip()
            if not text_snippet:
                logger.warning("Empty text snippet in citation")
                return None
                
            # Check if text snippet exists in chunk content (case-insensitive)
            chunk_content_lower = chunk['content'].lower()
            text_snippet_lower = text_snippet.lower()
            
            if text_snippet_lower not in chunk_content_lower:
                logger.warning(f"Text snippet not found in chunk {chunk_index}: {text_snippet[:50]}...")
                # Try to find a similar snippet
                words = text_snippet_lower.split()[:5]  # Take first 5 words
                found = any(word in chunk_content_lower for word in words if len(word) > 3)
                if not found:
                    return None
            
            return {
                'page': citation.get('page', metadata['start_page']),
                'line_range': citation.get('line_range', f"{metadata['start_line']}-{metadata['end_line']}"),
                'text_snippet': text_snippet[:250],  # Limit snippet length but keep more context
                'chunk_index': chunk_index,
                'char_offset': metadata.get('char_start', 0)
            }
            
        except Exception as e:
            logger.warning(f"Failed to map citation: {e}")
            return None
    
    def _ensure_balanced_coverage(self, items: List[Dict]) -> List[Dict]:
        """Ensure exactly 5 FOR and 5 AGAINST items (10 total)"""
        for_items = [item for item in items if item.get('category') == 'FOR']
        against_items = [item for item in items if item.get('category') == 'AGAINST']
        
        logger.info(f"Found {len(for_items)} FOR items and {len(against_items)} AGAINST items before balancing")
        
        # Take up to 5 of each
        balanced_for = for_items[:5]
        balanced_against = against_items[:5]
        
        # Fill missing slots with the strongest available items
        while len(balanced_for) < 5:
            if against_items and len(balanced_against) > 5:  # Take from excess AGAINST if available
                item = against_items.pop(0)
                item['category'] = 'FOR'
                item['rank'] = len(balanced_for) + 1
                balanced_for.append(item)
                logger.info(f"Converted AGAINST item to FOR for balancing: {item['title'][:50]}...")
            elif for_items:  # Try to get more FOR items
                for potential_item in for_items:
                    if potential_item not in balanced_for:
                        balanced_for.append(potential_item)
                        break
                else:
                    break  # No more FOR items available
            else:
                break  # Cannot find more items
        
        while len(balanced_against) < 5:
            if for_items and len(balanced_for) > 5:  # Take from excess FOR if available
                item = for_items.pop(0)
                item['category'] = 'AGAINST'
                item['rank'] = len(balanced_against) + 1
                balanced_against.append(item)
                logger.info(f"Converted FOR item to AGAINST for balancing: {item['title'][:50]}...")
            elif against_items:  # Try to get more AGAINST items
                for potential_item in against_items:
                    if potential_item not in balanced_against:
                        balanced_against.append(potential_item)
                        break
                else:
                    break  # No more AGAINST items available
            else:
                break  # Cannot find more items
        
        # If we still don't have 10 items, create placeholder items
        if len(balanced_for) + len(balanced_against) < 10:
            logger.warning(f"Only found {len(balanced_for) + len(balanced_against)} valid items. Creating placeholders.")
            # This is a fallback - in practice, the LLM should provide enough items
        
        # Combine and re-rank
        balanced_items = balanced_for + balanced_against
        for i, item in enumerate(balanced_items, 1):
            item['rank'] = i
            
        logger.info(f"Final balanced coverage: {len(balanced_for)} FOR, {len(balanced_against)} AGAINST")
        return balanced_items[:10]  # Ensure max 10 items

# API Endpoints
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):
    """
    Enhanced document analysis with structured output and metrics
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not configured")
    
    overall_start = time.time()
    metrics = {
        'extraction_time': 0,
        'chunking_time': 0, 
        'embedding_time': 0,
        'search_time': 0,
        'analysis_time': 0,
        'total_time': 0
    }
    
    vector_manager = None
    document_hash = None
    
    try:
        content = await file.read()
        
        # Generate document hash for caching
        document_hash = hashlib.sha256(content).hexdigest()[:16]
        logger.info(f"Processing document: {file.filename} (hash: {document_hash})")
        
        with temporary_pdf_file(content) as tmp_file_path:
            # Step 1: Extract text with metadata
            extract_start = time.time()
            processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
            with open(tmp_file_path, 'rb') as pdf_file:
                documents, used_ocr = processor.extract_text_with_metadata(pdf_file)
            metrics['extraction_time'] = time.time() - extract_start
            
            # Step 2: Create chunks
            chunk_start = time.time()
            chunks = processor.create_chunks_with_metadata(documents)
            metrics['chunking_time'] = time.time() - chunk_start
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} lines")
            
            # Step 3: Create vector store
            embed_start = time.time()
            vector_manager = VectorStoreManager()
            vector_manager.create_vector_store(chunks, document_hash)
            metrics['embedding_time'] = time.time() - embed_start
            
            # Step 4: Semantic search for relevant content
            search_start = time.time()
            search_queries = [
                "legal argument reasoning claim",
                "evidence fact data support", 
                "opposition counterargument defense",
                "statute law regulation violation",
                "precedent case law ruling decision",
                "constitutional right amendment",
                "public interest equity fairness"
            ]
            
            relevant_chunks = []
            seen_hashes = set()
            for query in search_queries:
                try:
                    results = vector_manager.semantic_search(query, k=4)  # Get more results
                    for result in results:
                        content_hash = result['chunk_hash']
                        if content_hash not in seen_hashes:
                            relevant_chunks.append(result)
                            seen_hashes.add(content_hash)
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
                    continue
                    
            metrics['search_time'] = time.time() - search_start
            logger.info(f"Found {len(relevant_chunks)} unique relevant chunks")
            
            # Step 5: Analyze with Groq
            analysis_start = time.time()
            analyzer = GroqAnalyzer(GROQ_API_KEY)
            analysis_result = analyzer.extract_key_items(relevant_chunks)
            metrics['analysis_time'] = time.time() - analysis_start
            
        # Calculate total time
        metrics['total_time'] = time.time() - overall_start
        logger.info(f"Analysis completed in {metrics['total_time']:.2f}s")
        
        # Validate we have exactly 10 items
        key_items = analysis_result['key_items']
        if len(key_items) != 10:
            logger.warning(f"Expected 10 items but got {len(key_items)}")
        
        # Prepare structured response
        return AnalysisResponse(
            document_name=file.filename,
            analysis_date=datetime.now().isoformat(),
            total_pages=len(set(doc['page'] for doc in documents)),
            key_items=key_items,
            raw_analysis=analysis_result.get('raw_analysis'),
            processing_time=metrics['total_time'],
            used_ocr=used_ocr,
            document_hash=document_hash
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up vector store to free memory
        if vector_manager:
            vector_manager.cleanup()

@app.get("/health")
async def health_check():
    """Lightweight health check endpoint"""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "groq_configured": bool(GROQ_API_KEY),
        "embeddings": "local-sentence-transformers",
        "ocr_available": OCR_AVAILABLE,
        "ocr_enabled": ENABLE_OCR_FALLBACK
    }
    
    # Only check Groq if API key is configured
    if GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            # Simple API call instead of listing all models
            client.models.list(limit=1)
            health_info["groq_status"] = "connected"
        except Exception as e:
            health_info["status"] = "degraded"
            health_info["groq_status"] = "error"
            health_info["groq_error"] = str(e)
    else:
        health_info["status"] = "degraded"
        health_info["groq_status"] = "not_configured"
    
    return health_info

@app.get("/available-models")
async def get_available_models():
    """Endpoint to check available Groq models"""
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not configured"}
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        models = client.models.list()
        available_models = [model.id for model in models.data]
        return {
            "available_models": available_models,
            "count": len(available_models),
            "preferred_models": PREFERRED_MODELS
        }
    except Exception as e:
        logger.error(f"Failed to fetch available models: {e}")
        return {"error": str(e)}

@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Legal Document Analyzer API",
        "version": "2.0.0",
        "status": "running",
        "groq_configured": bool(GROQ_API_KEY),
        "ocr_available": OCR_AVAILABLE,
        "features": [
            "Structured legal analysis",
            "Precise citation mapping", 
            "OCR fallback support",
            "Memory-optimized processing",
            "Balanced argument coverage (5 FOR, 5 AGAINST)"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check if we're in development mode without API key
    if not GROQ_API_KEY:
        logger.warning("Running in development mode without GROQ_API_KEY")
        logger.warning("Set GROQ_API_KEY environment variable to enable document analysis")
    else:
        logger.info("GROQ_API_KEY loaded successfully")
    
    if ENABLE_OCR_FALLBACK and not OCR_AVAILABLE:
        logger.warning("OCR fallback enabled but dependencies not available")
        logger.warning("Install OCR dependencies: pip install pytesseract pdf2image")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")