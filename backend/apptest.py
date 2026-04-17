from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain_pinecone import Pinecone as PineconeLangChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# ==== App & Env Setup ====
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Pinecone config
PINECONE_API_KEY = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMg"
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged"

# ==== Initialize Pinecone client ====
pc = Pinecone(api_key=PINECONE_API_KEY)

# ==== Embedding model ====
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==== Vectorstore ====
vectorstore = PineconeLangChain(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
    pinecone_api_key=PINECONE_API_KEY
)

# ==== LLM Setup: Local Mistral Model ====
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.5
)

llm = HuggingFacePipeline(pipeline=pipe)

# ==== QA Chain ====
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# ==== API Route ====
@app.route("/message", methods=["GET"])
def chat():
    user_query = request.args.get("text")
    if not user_query:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    try:
        response = qa_chain.invoke(user_query)
        return jsonify({
            "recipient_id": "user",
            "response": response
        })
    except Exception as e:
        print(" ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ==== Run App ====
if __name__ == "__main__":
    app.run(debug=True)
