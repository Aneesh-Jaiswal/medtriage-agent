"""
Builds and queries a Chroma vector store over the synthetic insurance policy
documents in rag/policy_docs/.

Usage:
    python rag/retriever.py --build      # (re)build the vector store
    python rag/retriever.py --query "does gold plan need auth for chest pain"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICY_DIR = Path(__file__).parent / "policy_docs"
PERSIST_DIR = Path(__file__).parent / "chroma_store"


def build_vector_store() -> Chroma:
    loader = DirectoryLoader(str(POLICY_DIR), glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    print(f"Built vector store with {len(chunks)} chunks -> {PERSIST_DIR}")
    return vector_store


def load_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=str(PERSIST_DIR), embedding_function=embeddings)


def get_retriever(k: int = 3):
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build the vector store")
    parser.add_argument("--query", type=str, help="Run a test query against the store")
    args = parser.parse_args()

    if args.build:
        build_vector_store()

    if args.query:
        retriever = get_retriever()
        results = retriever.invoke(args.query)
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:400])
