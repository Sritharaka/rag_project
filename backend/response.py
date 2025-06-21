import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_pinecone import Pinecone as LangchainPinecone
from pinecone import Pinecone


# Set environment variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMgw"

# Config variables
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#  Let langchain-pinecone handle Pinecone connection by just setting env variable
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

# HuggingFace text-generation model
pipeline_model = pipeline("text-generation", model="gpt2", temperature=0.5, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=pipeline_model)

# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce"
)

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/message", methods=["GET"])
def chat():
    user_query = request.args.get("text")
    if not user_query:
        return jsonify({"error": "Missing 'text' parameter"}), 400
    try:
        print("User query:", user_query)
        response = qa_chain.run(user_query)
        print("Answer:", response)
        return jsonify({"recipient_id": "user", "response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
