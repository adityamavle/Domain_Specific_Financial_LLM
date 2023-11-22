from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp
from paperqa import Docs
import paperscraper
# Make sure the model path is correct for your system!
# pull model from huggingface hub
llm = LlamaCpp(
    model_path="ggml-model-q4_0.bin", callbacks=[StreamingStdOutCallbackHandler()]
)
embeddings = LlamaCppEmbeddings(model_path="ggml-model-q4_0.bin")

docs = Docs(llm=llm, embeddings=embeddings)

keyword_search = 'bispecific antibody manufacture'
papers = paperscraper.search_papers(keyword_search, limit=2)
for path, data in papers.items():
    try:
        docs.add(path, chunk_chars=500)
    except ValueError as e:
        print('Could not read', path, e)

answer = docs.query(
    "What manufacturing challenges are unique to bispecific antibodies?")
print(answer)
