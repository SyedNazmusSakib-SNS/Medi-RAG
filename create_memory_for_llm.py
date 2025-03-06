# import libraries
import os
import sys
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# setp-1: load a raw pdf

DATA_path = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, 
                            glob = "*.pdf", 
                            loader_cls = PyPDFLoader)
    
    documents = loader.load()

    return documents

# step-2: create chunks

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, 
                                                chunk_overlap  = 50)
    
    chunks = text_splitter.split_documents(documents)

    return chunks

# step-3: create a vector embeddings

def get_embeddings_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    

    return embedding_model

embedding_model = get_embeddings_model()


# step-4: create a memory for llm


