import pickle
import time
import os
import langchain
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores.faiss import FAISS
load_dotenv()

def qaretriever(question,urls,st):
  main_loader = st.empty()
  llm = OpenAI(temperature = 0.9, max_tokens = 500)

  loaders = UnstructuredURLLoader(urls=urls)
  data = loaders.load()
  main_loader.text('Data Loading... ')
  text_splitter = RecursiveCharacterTextSplitter(separators=[
      '\n\n','\n',',','.',' '
  ],chunk_size=1000,chunk_overlap=200)
  main_loader.text('Recursive text splitter started Loading... ')
  docs = text_splitter.split_documents(data)
  embeddings = OpenAIEmbeddings()
  main_loader.text('...Embeddings started Loading... ')
  vector_index = FAISS.from_documents(docs, embeddings)
  file_path = "vector_index.pkl"
  with open(file_path, "wb") as f:
      pickle.dump(vector_index, f)
  if question:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
            result = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_index.as_retriever())
            output = result({"question":question},return_only_outputs=True)
            return output