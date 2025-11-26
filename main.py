"""Main application entry point.

This module provides CLI commands for the RAG system.
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv
import os

from src.ingest import ingest_documents
from src.chain import create_rag_chain
from src.query import interactive_query_loop, RAGQuery

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_ingest(args):
    """Handle ingest command."""
    logger.info(f"Indexing documents from {args.data_dir}")
    
    file_types = args.file_types.split(",") if args.file_types else None
    
    try:
        ingest_documents(
            data_dir=args.data_dir,
            persist_dir=args.vectorstore_dir,
            file_types=file_types,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print("\nIndexing completed successfully!")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        sys.exit(1)


def cmd_query(args):
    """Handle query command."""
    logger.info("Starting query interface")
    
    try:
        chain = create_rag_chain(
            vectorstore_path=args.vectorstore_dir,
            model_name=args.model,
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
    
    data_dir = Path(args.data_dir)
    vectorstore_dir = Path(args.vectorstore_dir)
    
    print(f"\n📁 Data Directory: {data_dir}")
    if data_dir.exists():
        files = list(data_dir.glob("*.*"))
        print(f"   Files: {len(files)}")
        for ext in [".txt", ".pdf", ".md"]:
            count = len(list(data_dir.glob(f"*{ext}")))
            if count > 0:
                print(f"   - {ext}: {count}")
    else:
        print("   ⚠️  Directory does not exist")
    
    print(f"\n💾 Vectorstore Directory: {vectorstore_dir}")
    if vectorstore_dir.exists():
        print("   Vectorstore exists")
    else:
        print("   ⚠️  Vectorstore not initialized (run 'ingest' first)")
    
    print(f"\n🤖 LLM Model: {args.model}")
    print(f"🔢 Embedding Model: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Demo - Professional RAG system with LangChain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Global arguments
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "./data"),
        help="Directory containing source documents"
    )
    parser.add_argument(
        "--vectorstore-dir",
        default=os.getenv("VECTORSTORE_DIR", "./vectorstore"),
        help="Directory for vector store"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model name"
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
        default=int(os.getenv("CHUNK_SIZE", 500)),
        help="Chunk size for splitting"
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 50)),
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
        default=int(os.getenv("TOP_K_DOCUMENTS", 3)),
        help="Number of documents to retrieve"
    )
    query_parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", 0.0)),
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
