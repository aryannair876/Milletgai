# 🌾 MilletsGAI — AI-Powered Millet Knowledge Assistant

**MilletsGAI** is an AI assistant that provides accurate, grounded information about Indian millets — covering nutrition, cultivation, recipes, health benefits, pest management, and market data. It combines a **fine-tuned Llama 3 8B model** with **Retrieval-Augmented Generation (RAG)** for factual, hallucination-resistant responses.

---

## ✨ Features

- **Fine-tuned Llama 3 8B** — SFT + DPO trained on a curated millet knowledge dataset
- **RAG Pipeline** — ChromaDB vector search for grounded, citation-backed answers
- **Chain-of-Thought Reasoning** — Automatic complexity detection triggers detailed step-by-step reasoning for complex queries
- **Smart Post-processing** — Cleans OCR artifacts, metadata leakage, and training artifacts from responses
- **Modern Web UI** — Next.js frontend with chat interface, confidence meters, and citation display
- **FastAPI Backend** — REST API with Fast/Thinking modes and model switching
- **Comprehensive Training Pipeline** — End-to-end scripts for data processing, SFT, and DPO training

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────┐
│  Next.js Frontend (Port 3000)   │
│  - Chat UI, Confidence Display  │
│  - Citation Panel               │
└──────────────┬──────────────────┘
               │ HTTP POST /chat
               ▼
┌─────────────────────────────────┐
│  FastAPI Backend (Port 8000)    │
│  - backend_api.py               │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  RAG Inference Engine           │
│  - rag_inference.py             │
│                                 │
│  1. Embed query (MiniLM-L6-v2) │
│  2. Semantic search (ChromaDB)  │
│  3. Build augmented prompt      │
│  4. Generate (Llama 3 + LoRA)   │
│  5. Post-process & clean        │
└─────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **NVIDIA GPU** with 8GB+ VRAM (for model inference)
- **CUDA 11.8+**
- ~20GB disk space for model weights

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/aryannair876/Milletgai.git
cd Milletgai
```

### 2. Python Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Download Model Weights

Download the following and place them in the project root:

| What | Where to Place | Source |
|---|---|---|
| **Llama 3 8B Instruct** | `Meta-Llama-3-8B-Instruct/` | [Meta on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| **MilletsGAI LoRA Adapter** | `models/milletsgai-dpo/` | *(trained locally — see Training section)* |

### 4. Build the Knowledge Base

```bash
python ingest_data.py
```

This creates the `chroma_db/` vector database from the training data.

### 5. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## ▶️ Running the Application

### Start the Backend

```bash
python backend_api.py
```

The FastAPI server starts on `http://localhost:8000`.

### Start the Frontend

```bash
cd frontend
npm run dev
```

The Next.js app starts on `http://localhost:3000`.

---

## 📁 Project Structure

```
milletai/
├── backend_api.py                 # FastAPI REST API server
├── rag_inference.py               # Core RAG + LLM inference engine
├── ingest_data.py                 # ChromaDB knowledge base builder
├── requirements.txt               # Python dependencies
├── accelerate_config.yaml         # HuggingFace Accelerate config
├── __init__.py
│
├── train_sft.py                   # Supervised Fine-Tuning script
├── train_dpo_only.py              # Direct Preference Optimization script
│
├── data/
│   ├── millets_data_final.jsonl   # Training dataset (instruction-output pairs)
│   ├── millets_dpo_final.jsonl    # DPO preference dataset
│   └── evaluation_questions.json  # Evaluation question set
│
├── src/
│   ├── data_processing/
│   │   ├── augment_dataset.py     # Data augmentation utilities
│   │   └── make_instruction_dataset.py  # Dataset creation from raw data
│   └── training/
│       ├── sft_train.py           # SFT training module
│       └── sft_train_windows.py   # Windows-compatible training
│
├── frontend/                      # Next.js web application
│   ├── src/
│   │   ├── app/                   # Next.js app router pages
│   │   ├── components/            # React UI components
│   │   └── lib/                   # Utility functions
│   ├── package.json
│   └── ...
│
├── Meta-Llama-3-8B-Instruct/     # [GITIGNORED] Base model weights
├── models/                        # [GITIGNORED] Fine-tuned adapters
└── chroma_db/                     # [GITIGNORED] Vector database
```

---

## 🧠 Training Your Own Model

### Step 1: Prepare Dataset

```bash
python src/data_processing/make_instruction_dataset.py
```

### Step 2: Supervised Fine-Tuning (SFT)

```bash
python train_sft.py
```

### Step 3: Direct Preference Optimization (DPO)

```bash
python train_dpo_only.py
```

The trained adapter will be saved to `models/milletsgai-dpo/`.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Meta Llama 3 8B Instruct |
| Fine-tuning | QLoRA (4-bit), SFT + DPO via HuggingFace TRL |
| RAG | LangChain + ChromaDB + all-MiniLM-L6-v2 |
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js 16, React 19, TailwindCSS 4, Framer Motion |
| Quantization | BitsAndBytes (4-bit NF4) |

---

## 📄 License

This project is for educational and research purposes.  
The base Llama 3 model is subject to the [Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).
