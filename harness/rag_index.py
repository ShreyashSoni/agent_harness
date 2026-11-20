"""CLI entry point: python -m harness.rag_index

Rebuilds the RAG index from the corpus directory. Run whenever the
corpus changes.
"""

from dotenv import load_dotenv
from harness.rag.indexer import build_index


def main() -> None:
    load_dotenv()
    build_index()


if __name__ == "__main__":
    main()