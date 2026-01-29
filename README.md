
# Domain_Specific_Financial_LLM

Domain-specific LLM system for **financial risk analysis and document-based question answering**, combining **fine-tuned finance models deployed via Ollama** with a lightweight **Streamlit** application and a supporting **RAG + finance data layer**.

- **Main application:** `stream_app.py`
- **Fine-tuning dataset:** https://huggingface.co/datasets/adityamavle/FinRiskAnalysis

---

## Overview

This project implements a unified LLM application for multiple **finance-specific NLP tasks**, including risk analysis, sentiment analysis, named entity recognition, and document Q&A.  
Models are **fine-tuned on a curated financial dataset**, registered with **Ollama**, and invoked through task-specific system prompts from a Streamlit UI.

Supported tasks:
- Risk Analysis
- Financial Sentiment Analysis
- Financial NER
- Financial Visual Data Analysis
- DocQA (document upload + retrieval-augmented QA)

---

## Architecture

### 1. Model Fine-Tuning
- Curated and fine-tuned domain-specific LLMs on the **FinRiskAnalysis** dataset.
- Task-specialized models (e.g., risk, sentiment, NER) uploaded and served via **Ollama**.

### 2. Application Layer
- `stream_app.py` provides a chat-style Streamlit interface.
- Requests are routed to the appropriate Ollama model based on the selected task.
- Responses are streamed token-by-token for interactive usage.

### 3. RAG and Finance Data Layer
- DocQA uses a supporting **RAG pipeline** for grounded answers over uploaded documents.
- A finance data layer handles document ingestion, preprocessing, and dataset access.
- RAG logic and data handling are kept modular and separate from UI code.

---

## Repository Structure

```
.
├── README.md
├── stream_app.py            # Main Streamlit application
├── doc.py                   # Document upload + DocQA entry point
├── rag_*                    # Retrieval-Augmented Generation utilities
├── finance_data_*           # Finance dataset loading / preprocessing
```

---

## How to Run

Install dependencies, then run the Streamlit application:

```bash
streamlit run stream_app.py
```

---

## Dataset

Fine-tuning dataset used in this project:

**FinRiskAnalysis (Hugging Face)**  
https://huggingface.co/datasets/adityamavle/FinRiskAnalysis

---

## Notes

- Task-specific model names (e.g., `mistral-risk`, `fin-sentiment`, `mistral-NER`) refer to fine-tuned Ollama models used by the app.
- Commit messages and timestamps in the repository are not part of the project description.
- The Streamlit app maintains conversational state across tasks using `st.session_state`.

---

## Links
- Dataset: https://huggingface.co/datasets/adityamavle/FinRiskAnalysis

