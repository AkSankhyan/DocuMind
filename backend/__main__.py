#!/usr/bin/env python3
"""
Command-line interface for the Agentic Document QA platform.
Allows indexing files and asking questions directly from terminal.
"""
import argparse
import os
import sys
import uuid
from .parsers import DocumentParser
from .vector_store import VectorStoreManager
from .agent import AgenticRAG


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Document QA CLI - Ingest documents and ask questions"
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to document file to index and query",
        required=True
    )
    parser.add_argument(
        "-q", "--question",
        help="Question to ask about the document",
        required=True
    )

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    print(f"\n[+] Parsing document: {args.file}...")
    doc_parser = DocumentParser()
    chunks, pages, _ = doc_parser.parse_and_chunk(args.file)
    print(f"[+] Extracted {len(chunks)} chunks across {pages} page(s).")

    collection_id = f"cli_{os.path.basename(args.file)}_{uuid.uuid4().hex[:6]}"
    
    print(f"[*] Indexing into Qdrant collection '{collection_id}'...")
    vs_manager = VectorStoreManager()
    vs_manager.index_chunks(collection_id, chunks)

    print(f"[*] Invoking LangGraph Corrective RAG Agent...")
    print(f"[?] Question: {args.question}\n")

    agent = AgenticRAG(vector_store=vs_manager)
    result = agent.run(question=args.question, collection_id=collection_id)

    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result.get("answer"))
    print("=" * 80)

    citations = result.get("citations", [])
    if citations:
        print("\nCITATIONS:")
        for idx, c in enumerate(citations, 1):
            print(f"[{idx}] Source: {c.get('source')}, Page {c.get('page')} (Score: {c.get('score', 0):.4f})")
            print(f"    Excerpt: {c.get('snippet')}\n")


if __name__ == "__main__":
    main()