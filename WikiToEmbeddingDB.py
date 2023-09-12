#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from langchain.document_loaders import PDFMinerLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from langchain.docstore.document import Document


#Load environment variables
persist_directory = "./KnowledgeDB"
source_directory = "./Knowledge"
model_path = "./llama.cpp/models/7B/ggml-model-q8_0.bin"
# embeddings_model_name = "text-embedding-ada-002"
chunk_size = 2000
chunk_overlap = 50
# openai_api_base = "https://xxxxx.openai.azure.com/" , openai_api_key = "xxxxxxxxxxxx", openai_api_type = "azure", deployment = "Your deployment name"


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    print(len(all_files))
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator=False)
    print(documents[1])
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    print(texts[1])
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def create_or_add_to_vectorDB():
    # Create embeddings
    # embeddings = OpenAIEmbeddings(openai_api_base=openai_api_base, openai_api_key=openai_api_key, openai_api_type=openai_api_type, deployment=deployment, chunk_size=chunk_size)
    embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=4096)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(texts.__len__())
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, chroma_db_impl='duckdb+parquet',anonymized_telemetry=False)
    db.persist()
    db = None




if __name__ == "__main__":
    create_or_add_to_vectorDB()
    # Test is vectorstore is working
    # embeddings = OpenAIEmbeddings(openai_api_base=openai_api_base, openai_api_key=openai_api_key, openai_api_type=openai_api_type, deployment=deployment, chunk_size=chunk_size)
    embeddings = LlamaCppEmbeddings(model_path=model_path, n_ctx=4096)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(db.similarity_search("What is Tokens?", k=1))