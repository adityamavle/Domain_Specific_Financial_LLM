import torch
from torch import cuda
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from huggingface_hub import login
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "adityamavle/llama-2-7b-fiqa"

cuda.is_available()
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
print(device)
access_token_read = "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"
login(token=access_token_read)

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Llama-2-7b-chat-hf', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
)


def generate_text(model_name, tokenizer, prompt):
    logging.set_verbosity(logging.CRITICAL)
    # Initialize the pipeline with the provided model and tokenizer
    pipe = pipeline(task="text-generation", model=model_name, tokenizer=tokenizer, max_length=1024)
    # Run the pipeline with the formatted prompt
    formatted_prompt = f"[INST] {prompt} [/INST]"
    result = pipe(formatted_prompt)
    # Print the prompt and the generated text
    print("This is the prompt")
    print(result[0]['generated_text'])


prompt = "Analyze the sentiment of the following sentence and answer in one word as 'positive', 'negative' or 'neutral': The GeoSolutions technology will leverage Benefon 's GPS solutions by providing Location Based Search Technology , a Communities Platform , location relevant multimedia content and a new and powerful commercial model"
generate_text(base_model, tokenizer,prompt)