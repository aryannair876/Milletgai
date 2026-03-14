"""
MilletsGAI - RAG Inference Script
==================================
This script combines Retrieval-Augmented Generation (RAG) with 
the fine-tuned Llama-3-8B model for accurate millet information.

HOW RAG WORKS (For Professor Explanation):
==========================================

Without RAG:
    User Query → Fine-Tuned Model → Response
    Problem: Model relies only on what it memorized during training

With RAG:
    User Query → [Vector Search] → Retrieved Documents → [Augmented Prompt] → Model → Response
    
The RAG Pipeline:
-----------------
1. QUERY VECTORIZATION:
   - Convert the user's question to a 384-dimensional vector
   - This is done by the same embedding model used during ingestion
   
2. SEMANTIC SEARCH (Similarity Search):
   - Find the K most similar documents from the vector database
   - Similarity is measured using cosine distance between vectors
   - "Most similar" = semantically related, not just keyword matching
   
3. CONTEXT AUGMENTATION:
   - Take the retrieved documents and add them to the prompt
   - The model now has relevant facts to ground its response
   
4. GENERATION:
   - The fine-tuned model generates a response using:
     a) Its learned millet knowledge (from fine-tuning)
     b) The specific facts provided via RAG context
   - This combination reduces hallucination significantly

WHY THIS IS BETTER:
-------------------
- Fine-tuning teaches the model HOW to talk about millets (style, format)
- RAG provides WHAT to say (grounded facts from the database)
- Together: accurate, well-formatted, fact-grounded responses

Dependencies:
    pip install langchain langchain-community sentence-transformers chromadb torch transformers peft bitsandbytes
"""

import os
import re
import sys
import time
import torch
from pathlib import Path
from typing import Tuple, List

# LangChain imports for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Transformers imports for the fine-tuned model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (relative to this script's location)
SCRIPT_DIR = Path(__file__).parent
CHROMA_PERSIST_DIR = SCRIPT_DIR / "chroma_db"
BASE_MODEL_PATH = SCRIPT_DIR / "Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = SCRIPT_DIR / "models" / "milletsgai-dpo"  # DPO model with enhanced friendly prompt

# Embedding model (must match what was used in ingest_data.py!)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Collection name in ChromaDB (must match ingest_data.py!)
COLLECTION_NAME = "millets_knowledge_base"

# RAG parameters
TOP_K_DOCUMENTS = 3  # Number of relevant documents to retrieve


# ============================================================================
# CHAIN OF THOUGHT (CoT) - Question Complexity Detection
# ============================================================================

# Keywords that indicate a complex question requiring step-by-step reasoning
COT_TRIGGER_PATTERNS = {
    # Comparison questions
    "compare": "comparison",
    "difference between": "comparison",
    "vs": "comparison", 
    "versus": "comparison",
    "better than": "comparison",
    "which is better": "comparison",
    "which millet": "comparison",
    "best millet": "comparison",
    
    # Health/Medical reasoning
    "diabetes": "health_reasoning",
    "diabetic": "health_reasoning",
    "blood sugar": "health_reasoning",
    "weight loss": "health_reasoning",
    "cholesterol": "health_reasoning",
    "heart health": "health_reasoning",
    "bone health": "health_reasoning",
    "anemia": "health_reasoning",
    "pregnant": "health_reasoning",
    "pregnancy": "health_reasoning",
    "child": "health_reasoning",
    "baby": "health_reasoning",
    "elderly": "health_reasoning",
    "disease": "health_reasoning",
    "condition": "health_reasoning",
    "should i eat": "health_reasoning",
    "can i eat": "health_reasoning",
    "is it safe": "health_reasoning",
    
    # Cultivation decisions
    "when to sow": "cultivation_decision",
    "best time to": "cultivation_decision",
    "which soil": "cultivation_decision",
    "how much water": "cultivation_decision",
    "irrigation": "cultivation_decision",
    "fertilizer": "cultivation_decision",
    "which region": "cultivation_decision",
    "yield": "cultivation_decision",
    "profit": "cultivation_decision",
    "market": "cultivation_decision",
    
    # Multi-factor questions
    "and also": "multi_factor",
    "as well as": "multi_factor",
    "both": "multi_factor",
    "multiple": "multi_factor",
}

def detect_question_complexity(question: str) -> tuple:
    """
    Detect if a question requires Chain of Thought reasoning.
    
    Complex questions include:
    - Comparisons (e.g., "Compare jowar and bajra")
    - Health-related decisions (e.g., "Best millet for diabetics")
    - Cultivation decisions (e.g., "When to sow pearl millet in Karnataka")
    - Multi-factor questions (e.g., "Which millet is high in calcium AND protein")
    
    Args:
        question: The user's question
        
    Returns:
        Tuple of (is_complex: bool, complexity_type: str or None)
    """
    import re
    question_lower = question.lower()
    
    # Check for trigger patterns
    for pattern, complexity_type in COT_TRIGGER_PATTERNS.items():
        if pattern in question_lower:
            return True, complexity_type
    
    # Check for question words that often indicate complex queries
    complex_starters = ["which", "what is the best", "how do i choose", "should i"]
    for starter in complex_starters:
        if question_lower.startswith(starter):
            return True, "decision_making"
    
    return False, None


# ============================================================================
# STEP 1: LOAD THE VECTOR DATABASE (For Retrieval)
# ============================================================================

def load_vector_database() -> Chroma:
    """
    Load the ChromaDB vector store created by ingest_data.py.
    
    This function:
    1. Initializes the same embedding model used during ingestion
    2. Connects to the persisted ChromaDB database
    3. Returns a vectorstore object ready for similarity search
    
    Returns:
        Chroma vectorstore object
    """
    print(f"\n📊 Loading vector database from: {CHROMA_PERSIST_DIR}")
    
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector database not found at {CHROMA_PERSIST_DIR}\n"
            "Please run ingest_data.py first to build the knowledge base!"
        )
    
    # Initialize embeddings (same model as ingestion!)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if GPU available
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load the existing ChromaDB collection
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    print(f"✅ Loaded vector database with collection: {COLLECTION_NAME}")
    return vectorstore


# ============================================================================
# STEP 2: LOAD THE FINE-TUNED LLAMA MODEL
# ============================================================================

def load_finetuned_model() -> Tuple:
    """
    Load the fine-tuned Llama-3-8B model with LoRA adapter.
    
    This uses 4-bit quantization (QLoRA) to reduce memory usage.
    The model is loaded in evaluation mode for inference.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n🧠 Loading fine-tuned model...")
    print(f"   Base model: {BASE_MODEL_PATH}")
    print(f"   LoRA adapter: {ADAPTER_PATH}")
    
    # Enable TF32 for faster computation on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_PATH),
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load LoRA adapter (fine-tuned weights)
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        is_trainable=False,
    )
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    return model, tokenizer


# ============================================================================
# STEP 3: SEMANTIC SEARCH (Retrieve Relevant Documents)
# ============================================================================

def retrieve_relevant_documents(vectorstore: Chroma, 
                                query: str, 
                                k: int = TOP_K_DOCUMENTS) -> List[Document]:
    """
    Perform semantic search to find the most relevant documents for a query.
    
    SEMANTIC SEARCH PROCESS:
    ------------------------
    1. The query text is converted to a 384-dimensional embedding vector
       using the all-MiniLM-L6-v2 model
       
    2. ChromaDB uses HNSW (Hierarchical Navigable Small World) algorithm
       to efficiently find the k nearest neighbors in vector space
       
    3. Cosine similarity is used as the distance metric:
       similarity = cos(θ) = (query_vec · doc_vec) / (|query_vec| * |doc_vec|)
       
    4. Documents are ranked by similarity and the top k are returned
    
    This is "semantic" because:
    - "jowar nutrition" matches "Sorghum nutritional profile"
    - "bajra cultivation" matches "Pearl millet growing conditions"
    - The model understands synonyms and related concepts!
    
    Args:
        vectorstore: ChromaDB vector store
        query: User's question
        k: Number of documents to retrieve
        
    Returns:
        List of most relevant Document objects
    """
    print(f"\n🔍 Performing semantic search for: \"{query[:50]}...\"")
    
    # similarity_search converts query to vector and finds k nearest neighbors
    results = vectorstore.similarity_search(query, k=k)
    
    print(f"   Retrieved {len(results)} relevant documents")
    for i, doc in enumerate(results, 1):
        millet = doc.metadata.get('millet_name', 'Unknown')
        category = doc.metadata.get('category', 'general')
        print(f"   {i}. [{category}] {millet}")
    
    return results


# ============================================================================
# STEP 4: BUILD THE AUGMENTED PROMPT (RAG Magic!)
# ============================================================================

def build_augmented_prompt(question: str, retrieved_docs: List[Document], 
                           use_cot: bool = False, complexity_type: str = None) -> str:
    """
    Construct an augmented prompt with retrieved context.
    
    Enhanced prompts that encourage concise, precise responses aligned with expected format.
    """
    
    # Format retrieved documents into context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        millet_name = doc.metadata.get('millet_name', 'Unknown')
        content = doc.page_content
        context_parts.append(f"About {millet_name}:\n{content}")
    
    context_string = "\n\n".join(context_parts)
    
    # Build prompts optimized for concise, precise responses
    if use_cot:
        # THINKING MODE: Detailed but structured
        augmented_prompt = f"""You are MilletsGAI, an expert on Indian millets.

TASK: Answer using ONLY the KNOWLEDGE below. Be precise and helpful.

RULES:
1. Start with the direct answer immediately
2. Include specific numbers (percentages, mg, g per 100g)
3. For recipes: list ingredients and key steps concisely
4. For comparisons: state the winner clearly with reason
5. Keep response focused and informative
6. Use friendly tone with occasional emojis 🌾

KNOWLEDGE:
{context_string}

QUESTION: {question}

ANSWER:"""
    else:
        # FAST MODE: Concise, direct answers
        augmented_prompt = f"""You are MilletsGAI, a friendly assistant for Indian millets 🌾

TASK: Give a concise, direct answer using the KNOWLEDGE below.

RULES:
1. Answer in 1-3 sentences when possible
2. Include exact values (7.3g protein, 344mg calcium, etc.)
3. For yes/no questions: start with Yes/No, then brief explanation
4. For recipes: name + key ingredients + basic steps
5. Be friendly and helpful!

KNOWLEDGE:
{context_string}

QUESTION: {question}

ANSWER:"""
    
    return augmented_prompt




# ============================================================================
# STEP 5: GENERATE RESPONSE WITH FINE-TUNED MODEL
# ============================================================================

def generate_response(model, tokenizer, prompt: str, 
                      max_tokens: int = 600, 
                      temperature: float = 0.6) -> str:
    """
    Generate a response using the fine-tuned Llama model.
    
    Args:
        model: The fine-tuned model with LoRA adapter
        tokenizer: The tokenizer
        prompt: The augmented prompt with retrieved context
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        
    Returns:
        Generated response text
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response with parameters tuned for detailed outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            min_new_tokens=50,  # Ensure minimum response length
            temperature=temperature,  # Slightly higher for more natural responses
            top_p=0.9,  # Slightly higher for more variety
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,  # Slightly lower to allow natural flow
        )
    
    # Calculate the number of input tokens
    input_length = inputs.input_ids.shape[1]
    
    # Slice the output to get only the generated tokens
    generated_tokens = outputs[0][input_length:]
    
    # Decode only the generated response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Post-process: Cut off at any hallucinated Q&A patterns
    cutoff_patterns = ["QUESTION:", "\nQ:", "\n\nQ:", "User:", "\nUser", "KNOWLEDGE:", "\nKNOWLEDGE"]
    for pattern in cutoff_patterns:
        if pattern in response:
            response = response.split(pattern)[0].strip()
    
    # ========================================================================
    # AGGRESSIVE ASTERISK AND FORMATTING CLEANUP
    # ========================================================================
    import re
    
    # Step 1: Remove ALL asterisks (they shouldn't appear in final output)
    # Handle markdown bold/italic that didn't render
    response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # **text** -> text
    response = re.sub(r'\*([^*\n]+)\*', r'\1', response)    # *text* -> text
    
    # Remove asterisks at start of line (common artifact)
    response = re.sub(r'^\*+\s*', '', response, flags=re.MULTILINE)
    
    # Remove asterisks before colons, exclamations, or other punctuation
    response = re.sub(r'\*+(?=[:!?,.\)])', '', response)
    
    # Remove asterisks after words (word* -> word)
    response = re.sub(r'(\w)\*+', r'\1', response)
    
    # Remove any remaining standalone asterisks
    response = re.sub(r'\s\*\s', ' ', response)
    response = re.sub(r'\*', '', response)  # Nuclear option - remove all remaining asterisks
    
    # Step 2: Fix formatting
    # Remove MilletsGAI prefix
    response = re.sub(r'^MilletsGAI:?\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r'^Hi!?\s*I\'?m?\s*happy to help!?\s*', '', response, flags=re.IGNORECASE)
    
    # Remove leading colons or dashes
    response = re.sub(r'^\s*[:\-]\s*', '', response)
    
    # Ensure proper bullet formatting
    response = re.sub(r'•\s*', '• ', response)
    response = re.sub(r'^\s*-\s+', '• ', response, flags=re.MULTILINE)  # - item -> • item
    
    # IMPORTANT: Convert inline bullets to newlines (• item • item -> \n• item\n• item)
    response = re.sub(r'\s+•\s+', '\n• ', response)
    
    # Remove contradictory phrases (after saying "Winner:", don't say "choose both")
    response = re.sub(r'Recommendation:\s*Choose both!?\s*', '', response, flags=re.IGNORECASE)
    response = re.sub(r"Let'?s compare:\s*", '\n', response)
    
    # ========================================================================
    # REMOVE TRAINING ARTIFACTS THAT HURT BLEU/ROUGE SCORES
    # ========================================================================
    training_artifacts = [
        r"Please provide your response[!.]?[^\n]*",
        r"Type 'done' when ready[!.]?[^\n]*",
        r"Your turn[!.]?[^\n]*",
        r"I'll assist further if needed[!.]?[^\n]*",
        r"Here's my response:[^\n]*",
        r"Ask away[!.]?",
        r"Let's get started[!.]?",
        r"Let's chat[!.]?",
        r"Please provide your query[!.]?",
        r"Please provide your answer[^\n]*",
        r"Would you like some specific information\?[^\n]*",
        r"```python[^`]*```",
        r"```[^`]*```",
        r"# Ask away!",
        r"# Let's get started!",
        r"\(Note:[^)]*\)",
        r"\(Also,[^)]*\)",
    ]
    
    for pattern in training_artifacts:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove trailing conversational phrases
    trailing_phrases = [
        r"Happy to help with anything else[!.]?\s*$",
        r"Please let me know if you need any further assistance[!.]?\s*$",
        r"Feel free to ask more[!.]?\s*$",
        r"Let me know if you have more questions[!.]?\s*$",
        r"What else would you like\?\s*$",
        r"Would you like any other recipe\?\s*$",
        r"Would you like any variations\?\s*$",
        r"Let's cook together[!.]?\s*$",
    ]
    
    for pattern in trailing_phrases:
        response = re.sub(pattern, '', response, flags=re.IGNORECASE)
    
    # Clean up spacing
    response = re.sub(r'  +', ' ', response)
    response = re.sub(r'\n +', '\n', response)
    response = re.sub(r'\n{3,}', '\n\n', response)  # Max 2 newlines
    response = response.strip()
    
    # Convert bullet characters to markdown-style for proper ReactMarkdown rendering
    response = re.sub(r'([.!?])\s*•\s*', r'\1\n• ', response)  # Sentence ending before bullet
    response = re.sub(r'\s+•\s+', '\n• ', response)  # Whitespace before bullet becomes newline
    response = re.sub(r'^•\s*', '- ', response, flags=re.MULTILINE)  # Convert bullet to dash at line start
    response = re.sub(r'\n•\s*', '\n- ', response)  # Convert bullet to dash after newline
    
                # Separate conclusions from list items - add newline before conclusion phrases
    # Use simpler string-based detection with regex
    conclusion_starters = [
        'So, if', 'So if', 'Therefore,', 'Need help', 'Would you like', 'Let me know', 
        "I'm here", "Here's your", "Here is your", "Now you can", "High Now",
        "For more details", "For more information", "Let's help", "Let's compare", 
        "Let's get cooking", "What do you need", "Hope this helps"
    ]
    for starter in conclusion_starters:
        # Special case for "High Now" -> "High. Now"
        if starter == "High Now":
            response = re.sub(r'(High)\s+(Now)', r'\1.\n\n\2', response)
        else:
            # If starter appears after non-whitespace, add newline before it
            escaped = re.escape(starter)
            response = re.sub(r'(\S)\s+(' + escaped + ')', r'\1\n\n\2', response)

    # Recipe Formatting Fix: If we see "1. Step" or "Step 1" inside a paragraph, break it out
    # Matches "text. 1. Step" -> "text.\n1. Step"
    response = re.sub(r'(\S)\s+(\d+\.\s+)', r'\1\n\2', response)
    
    # Recipe Fix: Replace " + " with newline if it acts as a separator (common in this model)
    # Only if it's surrounded by spaces and looks like a delimiter
    response = re.sub(r'\s+\+\s+', '\n- ', response) # " + " -> "\n- "

    # ========================================================================
    # RECIPE SEPARATION FIX: Put each recipe on its own line
    # ========================================================================
    # More aggressive: Any "Recipe X:" that follows ANY character (not newline) gets a newline
    # This handles cases like "Salt Recipe 3:" or "- 1 cup flour Recipe 2:"
    response = re.sub(r'(?<!\n)(Recipe\s*\d+\s*:)', r'\n\n\1', response)
    
    # Also handle just recipe names followed by colon (e.g., "Jowar Roti:" "Jowar Upma:")
    response = re.sub(r'(?<!\n)([A-Z][a-z]+\s+(?:Roti|Upma|Khichdi|Dosa|Idli|Ladoo|Laddu|Halwa|Porridge)\s*:)', r'\n\n\1', response)
    
    # Force "Ingredients:" to always start on new line
    response = re.sub(r'(?<!\n)(Ingredients?\s*:)', r'\n\1', response)
    
    # Force "Instructions:" / "Steps:" to always start on new line  
    response = re.sub(r'(?<!\n)(Instructions?\s*:)', r'\n\1', response)
    response = re.sub(r'(?<!\n)(Steps?\s*:)', r'\n\1', response)
    
    # Clean up any triple+ newlines that might result
    response = re.sub(r'\n{3,}', '\n\n', response)

    return response


# ============================================================================
# STEP 6: POST-PROCESSING (Same as original chatbot)
# ============================================================================

def clean_metadata_leakage(response: str) -> str:
    """
    Clean up metadata, disclaimers, and OCR artifacts that leak into responses.
    This is a post-processing step to remove unwanted content from the model output.
    """
    import re
    
    # Patterns to remove - aggressive cleaning for OCR artifacts and metadata
    removal_patterns = [
        # =========== ACKNOWLEDGEMENT/FOREWORD TEXT ===========
        # These are the main culprits for Ragi responses
        r'Thanks are due to scientists[^.]*\.',
        r'Thanks are due to[^.]*namely[^.]*\.',
        r'The support received from[^.]*\.',
        r'for data compilation[^.]*\.',
        r'Farmers FIRST project[^.]*\.',
        r'Ms\.?\s*Laxmi Prasanna[^.]*\.',
        r'SRF for data[^.]*\.',
        
        # Project staff names (standalone patterns)
        r'Laxmi Prasanna,?\s*SRF\s*Madhusudhana',
        r'Laxmi Prasanna,?\s*SRF',
        r'SRF\s*Madhusudhana',
        
        # ICAR Institute text
        r'The ICAR-Indian Institute of Millets Research[^.]*\.',
        r'ICAR-Indian Institute of Millets Research[^.]*\.',
        r'ABOUT THE INSTITUTE[^.]*\.',
        r'was established in \d{4}[^.]*\.',
        r'nodal centre for improving[^.]*\.',
        
        # Publication acknowledgements
        r'Thank you for your interest in our publication[^.]*\.',
        r'We hope that the information given in this bulletin[^.]*\.',
        r'will assist the extension personnel[^.]*\.',
        r'state department officials[^.]*\.',
        r'other millet workers[^.]*\.',
        r'This helps users quickly find[^.]*\.',
        r'For best results, consult local[^.]*\.',
        
        # Technical bulletin forewords/introductions
        r'The information contained in this technical bulletin[^.]*\.',
        r'would help to create the much needed awareness[^.]*\.',
        r'I appreciate the efforts put in by the scientists[^.]*\.',
        r'Lam confident that the information[^.]*\.',
        r'I am confident that the information[^.]*\.',
        r'assist the farmers and extension workers[^.]*\.',
        r'bringing out this bulletin[^.]*\.',
        r'in an impressive manner[^.]*\.',
        r'[Ff]oreword[^.]*\.',
        r'[Aa]cknowledgement[^.]*\.',
        
        # AICRP/Project metadata
        r'AICRP on Small Millets[^.]*\.',
        r'Project Co-?ordinator[^.]*\.',
        r'PzQo-\([^)]*\)',
        r'\(PRABHAKAR\)[^.]*\.',
        r'Date:\s*\d{1,2}-\d{1,2}-\d{4}[^.]*\.',
        r'Place:\s*Bengaluru[^.]*\.',
        r'during the year \d{4}[^.]*\.',
        
        # Disclaimer text (specific phrases only)
        r'This information is based on verified agricultural research and government sources\.?',
        r'Since,? these recommendations and advisory originate from the research system[^.]*\.',
        r'Any damage or loss resulting from the use of this book[^.]*\.',
        
        # =========== PUBLISHER INFO (Balajiscan) ===========
        r'Published by:\s*Director[^.]*INDIA\.?',
        r'Printed at:\s*Balajiscan[^.]*\.?',
        r'Cell:\s*\d+[^.]*\.?',
        r'Balajiscan Private Limited[^,]*,?',
        r'https?://[^\s]*balaji[^\s]*',  # URLs containing 'balaji'
        r'www\.[^\s]*balaji[^\s]*',
        r'Balaji[Ss]can[^.]*\.',
        r': https://www\.[^\s]+',  # Generic URL pattern at end of sentences
        
        # ISBN and broken tables
        r'\(ISBN:[^)]+\)',
        r'\|\s*Variety\s*\|\s*Region\s*\|+',
        r'\|\s*\|+\s*$',
        r'Choose the best variety based on your location[^.]*\.',
        
        # Generic acknowledgement patterns  
        r'farmers and extension personnel[^.]*\.',
        r'crop growing conditions, and plant protection technologies[^.]*\.',
        r'on the latest released varieties[^.]*\.',
        r'increase foxtail millet production[^.]*\.',
        
        # =========== TRAINING ARTIFACTS ===========
        r'EXTRA CREDIT[^.]*\.',
        r'EXTRA CREDIT:[^\n]*',
        r'Please provide accurate information[^.]*\.',
        r'Please feel free to ask[^.]*\.',
        r'Thanks for chatting[^.]*\.',
        
        # Additional OCR artifacts
        r'About Proso Millet[^:]*:',
        r'\* Highest fibre content among cereals',
    ]
    
    cleaned = response
    for pattern in removal_patterns:
        # Use re.IGNORECASE but NOT re.DOTALL (so .* doesn't match newlines)
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up multiple spaces (but preserve newlines for formatting)
    cleaned = re.sub(r'  +', ' ', cleaned)  # Multiple spaces to single
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
    cleaned = re.sub(r'^\s*[\*\-]\s*$', '', cleaned, flags=re.MULTILINE)  # Empty bullet points
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Clean multiple blank lines again
    cleaned = cleaned.strip()
    
    # If response becomes too short after cleaning, return something useful
    if len(cleaned) < 20:
        return "I couldn't find specific information about that in my knowledge base. Could you rephrase your question?"
    
    return cleaned


def correct_species_errors(response: str) -> str:
    """Post-processing to fix systematic species identification errors."""
    import re
    
    corrections = {
        r'barnyard\s+millet\s*(?:is|are|=|,)?\s*(?:also\s+)?(?:known\s+as|called|the\s+same\s+as)?\s*(?:chena|proso\s+millet)': 
            'Barnyard millet (Echinochloa species) is DIFFERENT from Chena/Proso millet (Panicum miliaceum)',
        r'pearl\s+millet\s+(?:has|contains|is\s+known\s+for)?\s*(?:the\s+)?highest\s+calcium':
            'Finger Millet (Ragi) has the highest calcium content among millets (~350mg per 100g)',
    }
    
    corrected = response
    for pattern, replacement in corrections.items():
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    
    return corrected


def filter_irrelevant_content(response: str, question: str) -> str:
    """
    Filter out content that doesn't match the question topic.
    For cooking questions, remove farming/pest content.
    For nutrition questions, remove drought/cultivation content.
    """
    import re
    
    question_lower = question.lower()
    
    # Detect question types
    is_cooking_question = any(kw in question_lower for kw in [
        'recipe', 'cook', 'make', 'prepare', 'dosa', 'idli', 'roti', 'porridge',
        'khichdi', 'upma', 'ladoo', 'laddu', 'halwa', 'eat', 'food', 'dish',
        'breakfast', 'lunch', 'dinner', 'snack', 'bake', 'fry'
    ])
    
    is_nutrition_question = any(kw in question_lower for kw in [
        'protein', 'fiber', 'fibre', 'calcium', 'iron', 'zinc', 'magnesium',
        'nutrition', 'nutritional', 'nutrient', 'calorie', 'energy', 'vitamin',
        'glycemic', 'carbohydrate', 'mineral'
    ])
    
    cleaned = response
    
    if is_cooking_question:
        # Remove farming/pest-related sentences from cooking responses
        farming_patterns = [
            r'[^.]*\b(?:inoculat|thiram|carbendazim|fungicide|pesticide|insecticide)\b[^.]*\.\s*',
            r'[^.]*\b(?:caterpillar|larvae|pest|insect pest|borer|aphid|mite)\b[^.]*\.\s*',
            r'[^.]*\b(?:seed treatment|soilborne disease|vector population)\b[^.]*\.\s*',
            r'[^.]*\b(?:sudan grass|johnson grass|alternate host)\b[^.]*\.\s*',
            r'[^.]*\bvegetative phase\b[^.]*\bdefoliation\b[^.]*\.\s*',
            r'[^.]*\b(?:sowing|transplanting|irrigation schedule|fertilizer application)\b[^.]*\.\s*',
            r'[^.]*\b(?:yield per hectare|crop rotation|field preparation)\b[^.]*\.\s*',
        ]
        
        for pattern in farming_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    if is_nutrition_question:
        # Remove drought tolerance and farming content from nutrition responses
        irrelevant_patterns = [
            r'[^.]*\bdrought\s+tolerance\s+is\s+(?:very\s+)?(?:high|low|medium)\b[^.]*\.\s*',
            r'[^.]*\bbased\s+on\s+these\s+ratings[^.]*\.\s*',
            r'[^.]*\bwater-scarce\s+conditions\b[^.]*\.\s*',
            r'[^.]*\b(?:sowing|transplanting|irrigation|cultivation)\b[^.]*\.\s*',
            r'\*\*[^*]+:\*\*\s*Drought\s+tolerance[^.]*\.\s*',
        ]
        
        for pattern in irrelevant_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up resulting formatting issues
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned


def fix_ocr_artifacts(response: str) -> str:
    """
    Fix OCR artifacts specifically for nutrition data.
    Handles patterns like '_protein per **' where values are missing.
    """
    import re
    
    # Fix patterns like "**_protein per **" or "**contains **_protein**"
    # These indicate missing/corrupted data
    response = re.sub(r'\*\*_+[^*]*\*\*', '**[data unavailable]**', response)
    response = re.sub(r'\*\*contains \*\*_+', '**contains approximately', response)
    response = re.sub(r'_protein per \*\*', 'protein per 100g**', response)
    
    # Clean up orphaned underscores around nutritional values
    response = re.sub(r'(\d+\.?\d*)\s*_+\s*(g|mg|%)', r'\1\2', response)
    response = re.sub(r'_+(\d+\.?\d*)\s*(g|mg|%)', r'\1\2', response)
    
    # Fix broken bold patterns
    response = re.sub(r'\*\*\s*\*\*', '', response)  # Empty bold
    response = re.sub(r'\*\*_+\*\*', '', response)  # Bold underscores only
    
    return response


def validate_millet_response(response: str, question: str) -> str:
    """
    Validate that the response mentions the correct millet.
    If a question asks about Jowar but response talks about Kuttu/Buckwheat,
    we need to flag this.
    """
    import re
    
    # Map of millet names to their incorrect alternatives that might appear
    millet_confusions = {
        # If question has these...   the response should NOT have these
        'jowar': ['kuttu', 'buckwheat', 'kutki', 'little millet'],
        'sorghum': ['kuttu', 'buckwheat', 'kutki', 'little millet'],
        'bajra': ['kuttu', 'buckwheat', 'ragi', 'finger millet'],
        'pearl millet': ['kuttu', 'buckwheat', 'ragi', 'finger millet'],
        'ragi': ['kuttu', 'buckwheat', 'jowar', 'sorghum'],
        'finger millet': ['kuttu', 'buckwheat', 'jowar', 'sorghum'],
        'kangni': ['kodo', 'barnyard', 'proso'],
        'foxtail': ['kodo', 'barnyard', 'proso'],
    }
    
    question_lower = question.lower()
    response_lower = response.lower()
    
    # Check if question asks about a specific millet
    for millet, wrong_millets in millet_confusions.items():
        if millet in question_lower:
            # Check if response mentions wrong millets prominently
            for wrong in wrong_millets:
                # Check for recipe names containing wrong millet
                if f'{wrong} puri' in response_lower or f'{wrong} roti' in response_lower:
                    # This is a recipe with wrong millet - flag it
                    print(f"⚠️ Millet validation: Question about {millet} but response mentions {wrong} recipe!")
                    # Replace with a generic message
                    return f"I apologize, but I couldn't find a specific recipe for **{millet.title()}** in my knowledge base. However, {millet.title()} can be used to make rotis, dosas, porridges, and various traditional dishes. Would you like me to provide general cooking instructions?"
                
                # Check if wrong millet appears in first 100 chars (too prominent)
                if wrong in response_lower[:100]:
                    print(f"⚠️ Millet validation: Question about {millet} but response leads with {wrong}!")
    
    return response


def add_source_citation(response: str) -> str:

    """Add a simple source note. Frontend already shows source badges."""
    # Keep it minimal - the frontend shows detailed source badges
    return response


# ============================================================================
# MAIN RAG PIPELINE
# ============================================================================

def rag_query(question: str, 
              vectorstore: Chroma, 
              model, 
              tokenizer,
              k: int = TOP_K_DOCUMENTS,
              max_tokens: int = 600,
              temperature: float = 0.3,
              use_adapter: bool = True,
              force_cot: bool = None) -> dict:  # Changed return annotation to dict to match implementation
    """
    Complete RAG pipeline: Retrieve → Augment → Generate
    
    This is the main function that ties everything together.
    Now includes Chain of Thought (CoT) reasoning for complex questions!
    
    Args:
        question: User's question
        vectorstore: ChromaDB vector store
        model: Fine-tuned Llama model
        tokenizer: Tokenizer
        k: Number of documents to retrieve
        max_tokens: Maximum response length
        temperature: Generation temperature
        use_adapter: Whether to use LoRA adapter (True) or Base model (False)
        
    Returns:
        Final response with post-processing applied
    """
    print(f"\n{'='*60}")
    print(f"🌾 RAG QUERY: {question}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Step 0: Detect if this question requires Chain of Thought reasoning
    # force_cot overrides auto-detection: True=always CoT, False=never CoT, None=auto
    if force_cot is True:
        is_complex = True
        complexity_type = "user_selected_thinking"
        print(f"🧠 Thinking mode enabled by user")
    elif force_cot is False:
        is_complex = False
        complexity_type = None
        print(f"⚡ Fast mode enabled by user")
    else:
        is_complex, complexity_type = detect_question_complexity(question)
    
    if is_complex and force_cot is None:
        print(f"🧠 Complex question detected! Type: {complexity_type}")
        print(f"   → Using Chain of Thought reasoning")
    
    if is_complex:
        # Increase token limit for CoT responses (they need more space for reasoning)
        effective_max_tokens = min(max_tokens, 600)
    else:
        print(f"📝 Simple question - using standard response")
        effective_max_tokens = min(max_tokens, 350)
    
    # Step 1: Retrieve relevant documents via semantic search
    retrieved_docs = retrieve_relevant_documents(vectorstore, question, k=k)
    
    # Step 2: Build augmented prompt with context (with CoT for complex questions)
    augmented_prompt = build_augmented_prompt(
        question, 
        retrieved_docs,
        use_cot=is_complex,  # Enable CoT for complex questions
        complexity_type=complexity_type
    )
    print(f"\n📝 Augmented prompt length: {len(augmented_prompt)} characters")
    
    # Step 3: Generate response with fine-tuned model
    # Step 3: Generate response with fine-tuned model
    print(f"\n🤖 Generating response (Adapter: {'ON' if use_adapter else 'OFF'})...")
    
    if use_adapter:
        response = generate_response(model, tokenizer, augmented_prompt, effective_max_tokens, temperature)
    else:
        # Use context manager to temporarily disable adapter
        # This requires 'model' to be a PeftModel
        try:
            with model.disable_adapter():
                response = generate_response(model, tokenizer, augmented_prompt, effective_max_tokens, temperature)
        except AttributeError:
             # Fallback if model is not a PeftModel
             print("⚠️ Model is not a PeftModel, cannot disable adapter via method. Using as is.")
             response = generate_response(model, tokenizer, augmented_prompt, effective_max_tokens, temperature)
    
    # Step 4: Post-processing pipeline
    response = clean_metadata_leakage(response)
    response = correct_species_errors(response)
    response = fix_ocr_artifacts(response)  # Fix corrupted nutrition data
    response = validate_millet_response(response, question)  # Validate correct millet
    response = filter_irrelevant_content(response, question)  # Topic-aware filtering
    response = add_source_citation(response)
    
    end_time = time.time()
    print(f"\n⏱️ Total time: {end_time - start_time:.2f} seconds")
    
    # Step 5: Separate thinking from answer for CoT responses
    # BUT: Don't show "thinking" for recipe questions - it's not useful
    thinking = None
    answer = response
    
    # Check if this is a recipe question - don't split thinking for recipes
    question_lower = question.lower()
    is_recipe_question = any(kw in question_lower for kw in [
        'recipe', 'cook', 'make', 'prepare', 'how to make', 'dish', 'food'
    ])
    
    if is_complex and not is_recipe_question:
        # Try to extract thinking portion from the response
        # Look for patterns that indicate reasoning vs final answer
        # Order matters - check more specific markers first
        answer_markers = [
            "Winner:",  # Most specific - indicates a clear conclusion
            "In conclusion",
            "My recommendation",
            "The answer is",
            "To summarize",
            "Overall,",
            "Therefore,",
        ]
        
        # Find if there's a clear answer marker
        best_split_idx = -1
        for marker in answer_markers:
            if marker.lower() in response.lower():
                idx = response.lower().find(marker.lower())
                if idx > 50:  # Ensure there's substantial thinking before
                    best_split_idx = idx
                    break
        
        # Only split if we found a good marker AND the answer would be substantial
        if best_split_idx > 0:
            potential_answer = response[best_split_idx:].strip()
            if len(potential_answer) > 80:  # Ensure answer is substantial
                thinking = response[:best_split_idx].strip()
                answer = potential_answer
            else:
                # Answer too short, keep full response as answer
                thinking = None
                answer = response
        else:
            # No clear marker found
            thinking = None
            answer = response
        
        # Clean up transition phrases from the answer (since thinking is shown separately)
        # Use regex to catch all variations like "based on:", "based on the above comparison;-", etc.
        transition_patterns = [
            r'^based\s+on[^:;.]*[:;.\-]+\s*',  # "based on...: " or "based on...;" or "based on...-"
            r'^from\s+the\s+above[^:;.]*[:;.\-]+\s*',  # "from the above...:"
            r'^so\s*,?\s*',  # "So," or "So "
            r'^\*\s*',  # Leading asterisk "* "
        ]
        for pattern in transition_patterns:
            match = re.match(pattern, answer, re.IGNORECASE)
            if match:
                answer = answer[match.end():].strip()
                # Capitalize first letter if needed
                if answer and answer[0].islower():
                    answer = answer[0].upper() + answer[1:]
                break
    
    # Clean up trailing underscores (OCR artifacts)
    answer = re.sub(r'_+([;,.])', r'\1', answer)  # Replace _; or _, or _. with just the punctuation
    answer = re.sub(r'\*\*([^*]+)_+\*\*', r'**\1**', answer)  # Fix **text_** to **text**
    answer = re.sub(r'_+\s*;', ';', answer)  # Clean _; patterns
    
    if thinking:
        thinking = re.sub(r'_+([;,.])', r'\1', thinking)
        thinking = re.sub(r'\*\*([^*]+)_+\*\*', r'**\1**', thinking)
        # Remove asterisks from thinking too
        thinking = re.sub(r'\*', '', thinking)
    
    # FINAL CLEANUP - Remove ALL asterisks from answer (nuclear option)
    answer = re.sub(r'\*', '', answer)
    # Clean up any double spaces after removals
    answer = re.sub(r'  +', ' ', answer)
    answer = answer.strip()
    
    # Return as dictionary with metadata
    return {
        "response": answer,
        "thinking": thinking,
        "is_complex": is_complex,
        "complexity_type": complexity_type if is_complex else None
    }


# ============================================================================
# INTERACTIVE CHAT LOOP
# ============================================================================

def interactive_chat():
    """
    Interactive chat interface for testing RAG + Fine-tuned model.
    """
    print("\n" + "="*60)
    print("🌾 MilletsGAI - RAG + Fine-Tuned Model Chat")
    print("="*60)
    print("Type 'quit' to exit, 'help' for sample questions\n")
    
    # Load components
    vectorstore = load_vector_database()
    model, tokenizer = load_finetuned_model()
    
    sample_questions = [
        "What are the nutritional benefits of finger millet?",
        "How do I cultivate pearl millet in drought conditions?",
        "What pests attack sorghum and how to control them?",
        "Give me a recipe for ragi mudde",
        "Which millet has the highest calcium content?",
        "What is the market price of foxtail millet?",
        "Compare the nutritional value of jowar and bajra"
    ]
    
    while True:
        print("\n" + "-"*40)
        question = input("❓ Your question: ").strip()
        
        if question.lower() == 'quit':
            print("\n👋 Thank you for using MilletsGAI!")
            break
        elif question.lower() == 'help':
            print("\n📝 Sample questions you can ask:")
            for i, q in enumerate(sample_questions, 1):
                print(f"   {i}. {q}")
            continue
        elif not question:
            print("⚠️ Please enter a question.")
            continue
        
        # Run RAG pipeline
        result = rag_query(question, vectorstore, model, tokenizer)
        
        print("\n" + "="*60)
        print("🤖 MilletsGAI Response:")
        print("="*60)
        
        # Display thinking if it was a complex question
        if result.get("thinking"):
            print("\n💭 Reasoning:")
            print(result["thinking"])
            print("\n" + "-"*40)
            print("\n📝 Answer:")
        
        print(result.get("response", ""))


# ============================================================================
# SINGLE QUERY MODE (For Integration)
# ============================================================================

class MilletsGAI_RAG:
    """
    Class wrapper for easy integration with existing chatbot.
    
    Usage:
        rag_system = MilletsGAI_RAG()
        response = rag_system.query("What is the calcium content in ragi?")
    """
    
    def __init__(self):
        """Initialize RAG components."""
        print("🌾 Initializing MilletsGAI RAG System...")
        self.vectorstore = load_vector_database()
        self.model, self.tokenizer = load_finetuned_model()
        print("✅ RAG System ready!")
    
    def query(self, question: str, 
              k: int = TOP_K_DOCUMENTS,
              max_tokens: int = 600,
              temperature: float = 0.3,
              use_adapter: bool = True,
              force_cot: bool = None) -> dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            k: Number of documents to retrieve (default: 3)
            max_tokens: Maximum response length (default: 600)
            temperature: Generation temperature (default: 0.3)
            use_adapter: Whether to use LoRA adapter (default: True)
            force_cot: Override CoT detection (True=always, False=never, None=auto)
            
        Returns:
            Dictionary with keys:
                - response: Main answer text
                - thinking: CoT reasoning (if complex question, else None)
                - is_complex: Whether question triggered CoT
                - complexity_type: Type of complexity detected
        """
        return rag_query(
            question=question,
            vectorstore=self.vectorstore,
            model=self.model,
            tokenizer=self.tokenizer,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            use_adapter=use_adapter,
            force_cot=force_cot
        )
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        Get similar documents without generating a response.
        Useful for debugging or showing retrieved context.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of relevant Document objects
        """
        return retrieve_relevant_documents(self.vectorstore, query, k=k)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode: python rag_inference.py "Your question here"
        question = " ".join(sys.argv[1:])
        vectorstore = load_vector_database()
        model, tokenizer = load_finetuned_model()
        result = rag_query(question, vectorstore, model, tokenizer)
        print("\n" + "="*60)
        print("🤖 Response:")
        print("="*60)
        
        # Display thinking if complex question
        if result.get("thinking"):
            print("\n💭 Reasoning:")
            print(result["thinking"])
            print("\n" + "-"*40)
        
        print(result.get("response", ""))
    else:
        # Interactive mode
        interactive_chat()
