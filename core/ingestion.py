import os
import fitz #Pymupdf
from typing import List, Dict, Any
from config.settings import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

#Interal Helpers

def _extract_pages(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append({
                "page_number": page_index + 1,
                "text": text 
            })

    doc.close()
    return pages

def _chunk_text(text: str, page_number: int, paper_name: str) -> List[Dict[str, Any]]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "paper_name": paper_name,
                "page_number": page_number,
            })    

        start += CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    return chunks

def ingest_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    paper_name = os.path.splitext(os.path.basename(pdf_path))[0]

    pages = _extract_pages(pdf_path)

    if not pages:
        raise ValueError(
            f"No extractable text found in '{paper_name}'."
             "The PDF may be scanned/image-only."
        )
    
    all_chunks: List[Dict[str, Any]] = []
    for page in pages:
        page_chunks = _chunk_text(
            text = page["text"],
            page_number = page["page_number"],
            paper_name = paper_name,
        )
        all_chunks.extend(page_chunks)

    return all_chunks

def ingest_multiple_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
   
    all_chunks: List[Dict[str, Any]] = []
 
    for path in pdf_paths:
        try:
            chunks = ingest_pdf(path)
            all_chunks.extend(chunks)
            print(f"  Ingested '{os.path.basename(path)}' -> {len(chunks)} chunks")
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipped '{os.path.basename(path)}': {e}")
 
    return all_chunks
 
