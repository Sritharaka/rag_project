from transformers import pipeline

llm = pipeline("text-generation", model="gpt2")  
response = llm("What is diabetes?", max_length=100)
print(response)
