#!/usr/bin/env python3
"""
Document QA Agent - Main Entry Point
Enterprise-grade Document Intelligence & Corrective RAG API Server.
"""

import sys
import os
from backend.api import run_api_server

if __name__ == "__main__":
    print("=" * 70)
    print(">> Starting Document Intelligence & Agentic RAG Platform...")
    print(">> Engine: LangGraph Corrective RAG + Qdrant Vector Store + FlashRank")
    print(">> Fast Inference via Groq LLM | Local Embeddings via FastEmbed")
    print(">> API running at: http://localhost:8000")
    print(">> Docs available at: http://localhost:8000/docs")
    print("=" * 70)
    
    run_api_server(host="0.0.0.0", port=8000, reload=False)