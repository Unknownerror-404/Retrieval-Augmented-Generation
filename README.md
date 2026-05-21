# Retrieval-Augmented-Generation

## Multilingual PDF Question-Answering Pipeline

### Overview

This project implements a **multilingual Retrieval-Augmented Generation (RAG)** pipeline for answering questions from PDF documents.

It supports:

- Japanese (`ja`)
- Vietnamese (`vi`)

The system:

1. Loads PDF files
2. Splits documents into chunks
3. Builds:
   - Dense vector embeddings (FAISS)
   - Sparse keyword search index (BM25)
4. Retrieves relevant chunks using hybrid search
5. Reranks results with a cross-encoder
6. Generates concise answers using an LLM
7. Writes predictions to a CSV submission file

---

## Architecture

```text

PDFs
  ↓
Chunking
  ↓
Embeddings (E5)
  ↓
FAISS Index + BM25
  ↓
Hybrid Retrieval
  ↓
Cross-Encoder Reranking
  ↓
LLM Answer Generation (Qwen)
  ↓
submission.csv
