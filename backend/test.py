import os
from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# Config
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_IfNGzzdUOOaVmiLcUCxJUgkjADOBYNOI"

PINECONE_API_KEY = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfM"
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"
PINECONE_REGION = "us-east-1"

# Initialize Pinecone client (v2)
pc = PineconeClient(api_key=PINECONE_API_KEY)

index = pc.Index(
    name=PINECONE_INDEX_NAME,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
)

# Embeddings object
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# LangChain Pinecone vectorstore - note param name is index=, NOT client=
vectorstore = Pinecone(
    index=index,
    embedding=embedding_model,
    text_key="text"
)

# LLM & chain
# llm = HuggingFaceEndpoint(
#     repo_id="gpt2",
#     temperature=0.5,
#     max_new_tokens=512
# )

# Create HuggingFace pipeline locally
pipeline_model = pipeline("text-generation", model="gpt2", temperature=0.5, max_new_tokens=512)

# Wrap it for LangChain
llm = HuggingFacePipeline(pipeline=pipeline_model)


qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

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
        # Test retrieval only
        query_vector = embedding_model.embed_query(user_query)
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        print("Retrieved:", results)
        
        # Generate answer
        response = qa_chain.run(user_query)
        print("Answer:", response)
        
        return jsonify({"recipient_id": "user", "response": response})
    except Exception as e:
        import traceback
        print(" ERROR:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
