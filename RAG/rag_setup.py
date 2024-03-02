import torch
from torch import cuda, bfloat16
import transformers
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import json
import os
from pprint import pprint
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

"""### Model Setup"""

MODEL_PATH = "llama-2-7b-finrisk"

device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
      MODEL_PATH,
      low_cpu_mem_usage=True,
      return_dict=True,
      torch_dtype=torch.float16,
      device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

"""### Text Generation Functions"""

def test_model(tokenizer, pipeline, prompt_to_test):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        prompt_to_test: the prompt
    Returns
        None
    """
    # adapted from https://huggingface.co/blog/llama2#using-transformers
    #time_1 = time()
    sequences = pipeline(
        prompt_to_test,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,)
    #time_2 = time()
    #print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    #for seq in sequences:
     #   print(f"Result: {seq['generated_text']}")

def generate_text(model_name, tokenizer, prompt):
    logging.set_verbosity(logging.CRITICAL)
    # Initialize the pipeline with the provided model and tokenizer
    pipe = transformers.pipeline(task="text-generation", model=model_name, tokenizer=tokenizer, max_length=1024)
    # Run the pipeline with the formatted prompt
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    result = pipe(formatted_prompt)
    # Print the prompt and the generated text
    print("This is the prompt")
    print(result[0]['generated_text'])

def gc1(collect_list):
  for i in range(len(collect_list)):
    del collect_list[i]
    gc.collect()

query_pipeline = transformers.pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens = 4000)
llm = HuggingFacePipeline(pipeline=query_pipeline)
# checking again that everything is working fine

"""### Document Loading"""

# Commented out IPython magic to ensure Python compatibility.
import locale
locale.getpreferredencoding = lambda: "UTF-8"
# %pip install --upgrade --quiet  "unstructured[all-docs]"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

from langchain_community.document_loaders import DirectoryLoader

# loader = loader = TextLoader("/content/rag_input.txt",encoding="utf8")
loader = DirectoryLoader('fin_documents', glob="**/*.pdf")
docs = loader.load()
print(f"{len(docs)} document(s) loaded.")

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

all_splits = text_splitter.split_documents(docs)
vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")


retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)


"""### Test RAG"""

def test_rag(qa, query):
    #generation_config = {"max_new_tokens": 100}
    print(f"Query: {query}\n")
    time_1 = time()
    result = qa.run(query)
    time_2 = time()
    print(f"Inference time: {round(time_2-time_1, 3)} sec.")
    print("\nResult: ", result)

query = "What is the document about?"
test_rag(qa, query)

# prompt: How to clear gpu memory of the prompt that I gave


#del query_pipeline
#gc.collect()
#gc.collect()
#gc.collect()
'''
query = "Mine a single paragraph from the exela document that exposes high liquidity risk."
test_rag(qa, query)

docs = vectordb.similarity_search("Mine a single paragraph from the exela document that exposes high liquidity risk.")
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print("Text: ", doc_details['page_content'], "\n")

docs = vectordb.similarity_search("Analyze exela's liquidity risk position")
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print("Text: ", doc_details['page_content'], "\n")

question = "Analyze exela's liquidity risk position based on its report"
result = qa({"query":question})
print(result["result"])
'''