import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_pinecone import Pinecone as LangchainPinecone
import traceback

# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_IfNGzzdUOOaVmiLcUCxJUgkjADOBYNOIE"
os.environ["PINECONE_API_KEY"] = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMg"

# Configuration
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"
MODEL_NAME = "gpt2"
MAX_TOTAL_TOKENS = 1024  # GPT-2 limit
MAX_NEW_TOKENS = 256     # Output length budget

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing Pinecone vector index
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

# Load Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define pipeline with explicit tokenizer control
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# QA chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
)

# Truncate function using tokenizer tokens
def truncate_text(text, max_tokens=MAX_TOTAL_TOKENS - MAX_NEW_TOKENS):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.convert_tokens_to_string(tokens)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/message", methods=["GET"])
def chat():
    user_query = request.args.get("text")
    if not user_query:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    try:
        # Truncate the user query tokens before passing to the model
        truncated_query = truncate_text(user_query)
        print(f"[User query truncated]: {truncated_query}")

        response = qa_chain.run(truncated_query)
        print(f"[Model response]: {response}")
        return jsonify({"recipient_id": "user", "response": response})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
