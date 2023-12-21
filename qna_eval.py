from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# Load the tokenizer and model from Hugging Face
model_name = "lmqg/t5-base-squad-qag"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate answers using T5
def generate_answer(question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the output sequence
    output_ids = model.generate(input_ids)[0]

    # Decode the output ids to a string
    answer = tokenizer.decode(output_ids, skip_special_tokens=True)
    return answer

# Example usage of the function
context = "Hugging Face is a company that provides tools for machine learning."
question = "What does Hugging Face provide?"
generated_answer = generate_answer(question, context)

# Reference answer for evaluation
reference_answer = "tools for machine learning"

# Calculate BLEU and METEOR scores
bleu_score = sentence_bleu([reference_answer.split()], generated_answer.split())
meteor_score = meteor_score([reference_answer], generated_answer)

print(f"Generated Answer: {generated_answer}")
print(f"BLEU Score: {bleu_score}")
print(f"METEOR Score: {meteor_score}")
