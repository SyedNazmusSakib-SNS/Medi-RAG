# import libraries
import os
import sys

from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate


# Setup LLM with HuggingFace

HF_token = os.getenv("HF_toekn")

huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.1"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(huggingface_repo_id = huggingface_repo_id, 
                            temperature= 0.5,
                            model_kwargs = {"token" : "HF_token", 
                                            "max_length" : "2048"})
    
    return llm


llm = load_llm(huggingface_repo_id)

# Connect LLM with FAISS memory

custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template = custom_prompt_template, 
                            input_variables = ["context", "question"])
    
    return prompt

# Create Chain