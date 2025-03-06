# setp-1: load a raw pdf
# step-2: create chunks
# step-3: create a vector embeddings
# step-4: create a memory for llm

import os
import sys
import json
import pandas as pd
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import SentenceTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter


