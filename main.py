
"""Main application entry point.

This module provides CLI commands for the RAG system.
"""

import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Disable ChromaDB telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import logging
import sys
from pathlib import Path

from src.config import AppConfig, resolve_path
from src.ingest import ingest_documents
from src.chain import create_rag_chain
from src.query import interactive_query_loop, RAGQuery
from src.logging_config import configure_logging

# Central logging configuration
app_config = AppConfig.load()
configure_logging(app_config.log_level)
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    """Handle ingest command."""
    data_dir = str(resolve_path(Path(args.data_dir)))
    vectorstore_dir = str(resolve_path(Path(args.vectorstore_dir)))
    logger.info(f"Indexing documents from {data_dir}")
    
    file_types = args.file_types.split(",") if args.file_types else None
    
    try:
        ingest_documents(
            data_dir=data_dir,
            persist_dir=vectorstore_dir,
            file_types=file_types,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
        )
        print("\nIndexing completed successfully!")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)


def cmd_query(args):
    """Handle query command."""
    logger.info("Starting query interface")
    
    try:
        vectorstore_dir = str(resolve_path(Path(args.vectorstore_dir)))
        chain = create_rag_chain(
            vectorstore_path=vectorstore_dir,
            model_name=args.model,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        
        if args.interactive:
            interactive_query_loop(chain, model_name=args.model)
        else:
            if not args.question:
                print("Error: --question is required for non-interactive mode")
                sys.exit(1)
            
            query_interface = RAGQuery(chain, model_name=args.model)
            response = query_interface.query(args.question, verbose=True)
            print(f"\n{response}")
            
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)


def cmd_info(args):
    """Display system information."""
    print("\n" + "="*60)
    print("RAG System Information")
    print("="*60)
    
    data_dir = resolve_path(Path(args.data_dir))
    vectorstore_dir = resolve_path(Path(args.vectorstore_dir))
    
    print(f"\n[DATA] Data Directory: {data_dir}")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        print(f"   Files: {len(files)}")
        for ext in [".txt", ".pdf", ".md"]:
            count = len(list(data_dir.glob(f"*{ext}")))
            if count > 0:
                print(f"   - {ext}: {count}")
    else:
        print("   [WARNING] Directory does not exist")
    
    print(f"\n[STORAGE] Vectorstore Directory: {vectorstore_dir}")
    if vectorstore_dir.exists():
        print("   [OK] Vectorstore exists")
    else:
        print("   [WARNING] Vectorstore not initialized (run 'ingest' first)")
    
    print(f"\n[MODEL] LLM Model: {args.model}")
    print(f"[EMBEDDING] Embedding Model: {args.embedding_model}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main CLI entry point."""
    config = app_config

    parser = argparse.ArgumentParser(
        description="RAG Demo - Professional RAG system with LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global arguments
    parser.add_argument(
        "--data-dir",
        default=str(config.data_dir),
        help="Directory containing source documents"
    )
    parser.add_argument(
        "--vectorstore-dir",
        default=str(config.vectorstore_dir),
        help="Directory for vector store"
    )
    parser.add_argument(
        "--model",
        default=config.model,
        help="Ollama model name"
    )
    parser.add_argument(
        "--embedding-model",
        default=config.embedding_model,
        help="Embedding model name"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Index documents into vectorstore")
    ingest_parser.add_argument(
        "--file-types",
        help="Comma-separated file types (e.g., txt,pdf,md)"
    )
    ingest_parser.add_argument(
        "--chunk-size",
        type=int,
        default=config.chunk_size,
        help="Chunk size for splitting"
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=config.chunk_overlap,
        help="Chunk overlap"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument(
        "-q", "--question",
        help="Question to ask (for non-interactive mode)"
    )
    query_parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive query loop"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=config.top_k_documents,
        help="Number of documents to retrieve"
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=config.temperature,
        help="LLM temperature"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display system information")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
