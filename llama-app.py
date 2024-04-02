from langchain_community.llms import Ollama
import time

llm = Ollama(model = 'llama2')

prompt = """
    Given the following paragraph, identify possible entities and classi
... fy them: To evaluate whether conditions and/or events raise substantial
... doubt about the company's ability to meet future financial obligations w
... ithin a year. Concerns include net losses, net operating cash outflows,
... working capital deficits, and significant interest payments on long-term
...  debt. Efforts to improve cash balances and liquidity over the next twel
... ve months are noted. Going concern discussions are in Note 2, focusing o
... n liquidity's role in meeting cash needs.
"""

start_time = time.time()
llm_response = llm.invoke(prompt)
end_time = time.time()

print("LLM response:", llm_response)
print("Time taken:", end_time - start_time, "seconds")