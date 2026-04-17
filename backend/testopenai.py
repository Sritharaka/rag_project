import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_openai import ChatOpenAI  # ✅ Use for OpenRouter or OpenAI

# ENV variables
os.environ["PINECONE_API_KEY"] = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMg"
os.environ["OPENAI_API_KEY"] = "sk-or-v1-4d89b4768eb5b6a0ed1391d03dcc3bebf7adb848e693dbec65a03ac721017af"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"  # ✅ Required if using OpenRouter

# Config
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"
MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4", "openrouter/mistralai/mixtral-8x7b-instruct"

# Step 1: Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Vectorstore
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

# Step 3: LLM - using OpenRouter or OpenAI
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.7,
    max_tokens=512,
    base_url=os.environ["OPENAI_API_BASE"],  # Important for OpenRouter
)

# Step 4: Retriever + QA Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# Step 5: Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route("/message", methods=["GET"])
def chat():
    user_query = request.args.get("text")
    if not user_query:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    try:
        print(f"[User Query]: {user_query}")
        response = qa_chain.invoke({"query": user_query})
        print(f"[Model Response]: {response['result']}")
        return jsonify({"recipient_id": "user", "response": response['result']})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
