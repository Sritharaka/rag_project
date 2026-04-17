from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "sk-..."  # Your OpenAI key here
PINECONE_API_KEY = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMg"
PINECONE_ENV = "us-east-1"  # You might not need this explicitly now
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged"

# Initialize Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Access your index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize LangChain vectorstore with Pinecone index
embedding_model = OpenAIEmbeddings()
vectorstore = PineconeLangChain(index, embedding_model, text_key="text")

# Initialize the LLM chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.route("/message", methods=["GET"])
def chat():
    user_query = request.args.get("text")
    if not user_query:
        return jsonify({"error": "Missing 'text' parameter"}), 400

    response = qa_chain.run(user_query)

    return jsonify({
        "recipient_id": "user",
        "response": response
    })

if __name__ == "__main__":
    app.run(debug=True)
