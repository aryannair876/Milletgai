"""
MilletsGAI - RAG Data Ingestion Script
=======================================
This script builds a ChromaDB vector database from the millet dataset.

SEMANTIC SEARCH EXPLAINED:
--------------------------
Traditional keyword search matches exact words. Semantic search understands MEANING.

How it works:
1. We convert each row of data into a text description (document)
2. An embedding model (all-MiniLM-L6-v2) converts text into a 384-dimensional vector
   - Similar meanings → vectors that are close together in 384D space
   - Different meanings → vectors that are far apart
3. ChromaDB stores these vectors in an indexed structure for fast retrieval
4. When a user asks a question:
   - The question is also converted to a vector
   - We find the K nearest vectors (documents) using cosine similarity
   - These are the most semantically relevant documents to the query

Example:
- Query: "What nutrients are in jowar?"
- Even if the database says "Sorghum nutritional profile", semantic search
  will match it because the MEANING is similar (jowar = sorghum)

Dependencies:
    pip install langchain langchain-community sentence-transformers chromadb pandas
"""

import os
import json
import pandas as pd
from pathlib import Path

# LangChain imports for document handling and vector stores
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your millet dataset (JSONL format)
# Using the cleaned data file which has the instruction-output pairs
DATA_PATH = Path(__file__).parent / "data" / "millets_data_final.jsonl"
# Additional data file with correct MSP/market information
MSP_DATA_PATH = Path(__file__).parent / "data" / "msp_market_data.jsonl"

# Directory where the vector database will be stored
CHROMA_PERSIST_DIR = Path(__file__).parent / "chroma_db"

# Embedding model - runs locally, no API key needed
# all-MiniLM-L6-v2 is a good balance of speed and quality (384 dimensions)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Collection name in ChromaDB
COLLECTION_NAME = "millets_knowledge_base"


# ============================================================================
# STEP 1: LOAD DATA FROM JSONL
# ============================================================================

def load_millet_data(data_path: Path) -> list[dict]:
    """
    Load millet data from JSONL file.
    
    Each line in the JSONL file contains:
    - instruction: The question or query
    - input: Optional additional context
    - output: The answer/information
    - meta: Metadata including millet_name and source
    
    Returns:
        List of dictionaries, each representing one Q&A pair
    """
    print(f"📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    documents = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                documents.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping malformed line {line_num}: {e}")
    
    print(f"✅ Loaded {len(documents)} instruction-output pairs")
    return documents


# ============================================================================
# STEP 2: CONVERT DATA ROWS TO TEXT CHUNKS (DOCUMENTS)
# ============================================================================

def create_documents_from_millet_data(raw_data: list[dict]) -> list[Document]:
    """
    Convert raw JSONL data into LangChain Document objects.
    
    SEMANTIC SEARCH KEY INSIGHT:
    ----------------------------
    The text we create here is EXACTLY what gets embedded and searched.
    We combine question + answer into a single chunk so that:
    1. Questions about "ragi nutrition" will match chunks about ragi nutrition
    2. The full context (Q + A) is retrieved together
    
    Each Document has:
    - page_content: The text that gets embedded (question + answer combined)
    - metadata: Extra info for filtering and citation (millet name, source, etc.)
    
    Args:
        raw_data: List of dicts from JSONL file
        
    Returns:
        List of LangChain Document objects ready for embedding
    """
    print("📝 Converting data rows to searchable documents...")
    
    documents = []
    
    for idx, item in enumerate(raw_data):
        # Extract fields with safe defaults
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        meta = item.get("meta", {})
        
        # Skip empty entries
        if not output:
            continue
        
        # ====================================================================
        # CREATE THE TEXT CHUNK (This is what gets embedded!)
        # ====================================================================
        # Format: Combine instruction and output so semantic search can match
        # queries that are similar to either the question OR the answer
        
        # If there's additional input context, include it
        if input_text:
            text_content = f"Question: {instruction}\nContext: {input_text}\n\nAnswer: {output}"
        else:
            text_content = f"Question: {instruction}\n\nAnswer: {output}"
        
        # ====================================================================
        # CREATE METADATA (For filtering and citations)
        # ====================================================================
        metadata = {
            "millet_name": meta.get("millet_name", "Unknown"),
            "millet_id": meta.get("millet_id", ""),
            "source": meta.get("source", "MilletsGAI Training Data"),
            "instruction": instruction,  # Keep original question for display
            "doc_index": idx
        }
        
        # Detect category from instruction text for filtering
        instruction_lower = instruction.lower()
        if any(kw in instruction_lower for kw in ["nutri", "protein", "vitamin", "calcium", "iron", "health"]):
            metadata["category"] = "nutrition"
        elif any(kw in instruction_lower for kw in ["cultiv", "grow", "soil", "irrigat", "climate", "season"]):
            metadata["category"] = "cultivation"
        elif any(kw in instruction_lower for kw in ["pest", "disease", "control", "ipm", "management"]):
            metadata["category"] = "pest_management"
        elif any(kw in instruction_lower for kw in ["market", "price", "cost", "value", "demand"]):
            metadata["category"] = "market"
        elif any(kw in instruction_lower for kw in ["recipe", "cook", "prepare", "dish"]):
            metadata["category"] = "recipe"
        else:
            metadata["category"] = "general"
        
        # Create LangChain Document object
        doc = Document(page_content=text_content, metadata=metadata)
        documents.append(doc)
    
    print(f"✅ Created {len(documents)} searchable documents")
    
    # Print category distribution
    from collections import Counter
    categories = Counter(doc.metadata["category"] for doc in documents)
    print("📊 Category distribution:")
    for cat, count in sorted(categories.items()):
        print(f"   - {cat}: {count}")
    
    return documents


# ============================================================================
# STEP 3: CREATE EMBEDDINGS AND BUILD VECTOR DATABASE
# ============================================================================

def build_vector_database(documents: list[Document], 
                          persist_dir: Path,
                          embedding_model: str,
                          collection_name: str) -> Chroma:
    """
    Build a ChromaDB vector store from documents.
    
    EMBEDDING PROCESS:
    ------------------
    1. Load the embedding model (all-MiniLM-L6-v2)
       - This is a neural network trained to map text → 384D vectors
       - Similar texts get similar vectors (close in vector space)
    
    2. For each document:
       - Pass the page_content through the embedding model
       - Get a 384-dimensional vector representation
       - Store the vector + original text + metadata in ChromaDB
    
    3. ChromaDB creates an HNSW index for fast approximate nearest neighbor search
       - This allows millisecond retrieval even with thousands of documents
    
    Args:
        documents: List of LangChain Document objects
        persist_dir: Where to save the ChromaDB database
        embedding_model: HuggingFace model name for embeddings
        collection_name: Name of the collection in ChromaDB
        
    Returns:
        Chroma vector store object (also saved to disk)
    """
    print(f"\n🧠 Loading embedding model: {embedding_model}")
    print("   (This will download ~90MB on first run)")
    
    # Initialize the embedding model
    # HuggingFaceEmbeddings wraps sentence-transformers for LangChain
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # Normalize for cosine similarity
    )
    
    print(f"\n📊 Building vector database with {len(documents)} documents...")
    print(f"   Persist directory: {persist_dir}")
    
    # Create the persist directory if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # CREATE THE CHROMA VECTOR STORE
    # ========================================================================
    # Chroma.from_documents():
    # 1. Takes each document's page_content
    # 2. Passes it through the embedding model
    # 3. Stores the resulting vector + text + metadata
    # 4. Builds an HNSW index for fast similarity search
    # 5. Persists everything to disk
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name
    )
    
    print(f"✅ Vector database built and saved!")
    print(f"   Collection: {collection_name}")
    print(f"   Documents indexed: {len(documents)}")
    
    return vectorstore


# ============================================================================
# STEP 4: TEST THE VECTOR DATABASE (Semantic Search Demo)
# ============================================================================

def test_semantic_search(vectorstore: Chroma, test_queries: list[str], k: int = 3):
    """
    Test semantic search with sample queries.
    
    SIMILARITY SEARCH EXPLAINED:
    ----------------------------
    When you call vectorstore.similarity_search(query, k=3):
    
    1. The query text is converted to a 384D vector using the same embedding model
    2. ChromaDB finds the k vectors in the database that are NEAREST to the query vector
       - Uses cosine similarity: cos(θ) = (A·B)/(|A||B|)
       - cos(θ) = 1.0 means identical, 0.0 means orthogonal (unrelated)
    3. Returns the documents corresponding to those nearest vectors
    
    This is "semantic" because:
    - "jowar nutrition" will match "Sorghum nutritional benefits" (same meaning!)
    - "ragi calcium content" will match "Finger millet has 344mg calcium per 100g"
    - It understands synonyms, paraphrases, and related concepts
    
    Args:
        vectorstore: The ChromaDB vector store
        test_queries: List of test queries to run
        k: Number of results to return per query
    """
    print("\n" + "="*60)
    print("🔍 SEMANTIC SEARCH TEST")
    print("="*60)
    
    for query in test_queries:
        print(f"\n📌 Query: \"{query}\"")
        print("-" * 50)
        
        # Perform similarity search
        # This converts the query to a vector and finds the k nearest documents
        results = vectorstore.similarity_search(query, k=k)
        
        for i, doc in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   📗 Millet: {doc.metadata.get('millet_name', 'N/A')}")
            print(f"   📁 Category: {doc.metadata.get('category', 'N/A')}")
            # Show first 200 chars of content
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"   📄 Preview: {content_preview}...")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to build the RAG knowledge base.
    
    Pipeline:
    1. Load JSONL data (instruction-output pairs)
    2. Convert each row to a LangChain Document
    3. Create embeddings using all-MiniLM-L6-v2
    4. Store in ChromaDB with persistence
    5. Run test queries to verify semantic search works
    """
    print("\n" + "="*60)
    print("🌾 MilletsGAI - RAG Knowledge Base Builder")
    print("="*60)
    
    # Step 1: Load main data
    raw_data = load_millet_data(DATA_PATH)
    
    # Step 1b: Load MSP/market data if available
    if MSP_DATA_PATH.exists():
        msp_data = load_millet_data(MSP_DATA_PATH)
        print(f"✅ Adding {len(msp_data)} MSP/market entries")
        raw_data.extend(msp_data)
    else:
        print(f"⚠️ MSP data file not found: {MSP_DATA_PATH}")
    
    print(f"📊 Total entries to index: {len(raw_data)}")
    
    # Step 2: Convert to documents
    documents = create_documents_from_millet_data(raw_data)
    
    # Step 3: Build vector database
    vectorstore = build_vector_database(
        documents=documents,
        persist_dir=CHROMA_PERSIST_DIR,
        embedding_model=EMBEDDING_MODEL,
        collection_name=COLLECTION_NAME
    )
    
    # Step 4: Test with sample queries
    test_queries = [
        "What is the calcium content in ragi?",
        "How to grow pearl millet in drought conditions?",
        "What pests attack jowar crops?",
        "Give me a recipe for bajra roti",
        "Which millet is best for diabetics?"
    ]
    
    test_semantic_search(vectorstore, test_queries, k=3)
    
    print("\n" + "="*60)
    print("✅ RAG KNOWLEDGE BASE READY!")
    print("="*60)
    print(f"\n📁 Database saved to: {CHROMA_PERSIST_DIR}")
    print("🚀 You can now run rag_inference.py to query your fine-tuned model with RAG!")


if __name__ == "__main__":
    main()
