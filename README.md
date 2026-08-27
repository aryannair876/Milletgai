# 🌾 MilletsGAI — AI-Powered Millet Knowledge Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![Model: Llama 3 8B](https://img.shields.io/badge/model-Llama%203%208B-orange.svg)](https://llama.meta.com/llama3/)

**MilletsGAI** is an AI assistant that provides accurate, grounded information about Indian millets — covering nutrition, cultivation, recipes, health benefits, pest management, and market data. It combines a **fine-tuned Llama 3 8B model** with **Retrieval-Augmented Generation (RAG)** for factual, hallucination-resistant responses.

---

## ✨ Features

- **Fine-tuned Llama 3 8B** — SFT + DPO trained on a curated millet knowledge dataset
- **RAG Pipeline** — ChromaDB vector search for grounded, citation-backed answers
- **Chain-of-Thought Reasoning** — Automatic complexity detection triggers detailed step-by-step reasoning for complex queries
- **Smart Post-processing** — Cleans OCR artifacts, metadata leakage, and training artifacts from responses
- **Modern Web UI** — Next.js frontend with chat interface, confidence meters, and citation display
- **FastAPI Backend** — REST API with Fast/Thinking modes and model switching
- **Reproducible Training Pipeline** — End-to-end scripts for data processing, SFT, and DPO

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
│  1. Embed query (MiniLM-L6-v2)  │
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

> Developed and tested on an RTX 4060 (8GB). Training configs are tuned for that VRAM budget.

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

Model weights are **not** included in this repository. Download them and place as follows:

| What | Where to Place | Source |
|---|---|---|
| **Llama 3 8B Instruct** | `Meta-Llama-3-8B-Instruct/` | [Meta on HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) |
| **MilletsGAI LoRA Adapter** | `models/milletsgai-dpo/` | *(trained locally — see [Training](#-training-your-own-model))* |

> Access to the base model requires accepting Meta's license on HuggingFace.

### 4. Build the Knowledge Base

```bash
python ingest_data.py
```

This creates the `chroma_db/` vector database from the training data.

### 5. Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local   # adjust NEXT_PUBLIC_API_URL if your backend isn't on :8000
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

### Query From the Command Line

You can also query the RAG engine directly, without the web stack:

```bash
python rag_inference.py "What is the protein content of finger millet?"
```

---

## 📁 Project Structure

```
milletai/
├── backend_api.py                 # FastAPI REST API server
├── rag_inference.py               # Core RAG + LLM inference engine (also a CLI)
├── ingest_data.py                 # ChromaDB knowledge base builder
├── train_sft.py                   # Stage 1: Supervised Fine-Tuning
├── train_dpo_only.py              # Stage 2: Direct Preference Optimization
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT (code only — see note on model weights)
│
├── data/
│   ├── millets_data_final.jsonl   # SFT dataset (instruction–output pairs)
│   ├── millets_dpo_final.jsonl    # DPO preference dataset
│   └── evaluation_questions.json  # Evaluation question set
│
├── src/
│   └── data_processing/
│       ├── make_instruction_dataset.py  # Build instruction dataset from raw sources
│       └── augment_dataset.py           # Data augmentation utilities
│
├── frontend/                      # Next.js web application
│   ├── src/
│   │   ├── app/                   # Next.js app router pages
│   │   ├── components/            # React UI components
│   │   └── lib/                   # Utility functions
│   ├── .env.example               # Frontend configuration template
│   └── package.json
│
├── Meta-Llama-3-8B-Instruct/      # [not in repo] Base model weights
├── models/                        # [not in repo] Fine-tuned LoRA adapters
└── chroma_db/                     # [not in repo] Vector database (built by ingest_data.py)
```

---

## 🧠 Training Your Own Model

The pipeline runs in three stages:

### Step 1: Prepare Dataset

```bash
python src/data_processing/make_instruction_dataset.py
```

### Step 2: Supervised Fine-Tuning (SFT)

```bash
python train_sft.py
```

Saves a LoRA adapter to `models/milletsgai-final/`.

### Step 3: Direct Preference Optimization (DPO)

```bash
python train_dpo_only.py
```

Loads the SFT adapter and applies preference tuning. The final adapter is saved to `models/milletsgai-dpo/` — this is what `rag_inference.py` loads at runtime.

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

## 📚 Citation

This work was presented as:

> **MilletsGAI: An LLM for the Indian Millet Ecosystem**
> Aryan Nair, Khushi Thakur, Subham Dey
> International Conference on Emerging Methodologies in Computing, Sciences and Informatics (ICEMCSI 2026),
> New Horizon College of Engineering, Bengaluru, India.

```bibtex
@inproceedings{nair2026milletsgai,
  title     = {MilletsGAI: An LLM for the Indian Millet Ecosystem},
  author    = {Nair, Aryan and Thakur, Khushi and Dey, Subham},
  booktitle = {Proceedings of the International Conference on Emerging Methodologies
               in Computing, Sciences and Informatics (ICEMCSI)},
  year      = {2026},
  address   = {Bengaluru, India}
}
```

---

## 📄 License

Source code in this repository is released under the [MIT License](LICENSE).

The **Meta Llama 3 8B** base model — and any adapter fine-tuned from it — remains subject to the
[Meta Llama 3 Community License](https://llama.meta.com/llama3/license/). Obtain the base weights
directly from Meta or HuggingFace and accept their terms before use.
