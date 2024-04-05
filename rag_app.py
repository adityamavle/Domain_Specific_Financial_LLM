import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama

data_path = "finance_data\Intel-10K-report.pdf"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=30,
    length_function=len,)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embedding_func = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(documents, embedding=embedding_func)

template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""
pt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

rag = RetrievalQA.from_chain_type(
            llm=Ollama(model="mistral:instruct"),
            retriever=vectordb.as_retriever(),
            memory=ConversationSummaryMemory(llm = Ollama(model="mistral:instruct")),
            chain_type_kwargs={"prompt": pt, "verbose": True},
        )

rag.invoke("What are Intel's main products ?")