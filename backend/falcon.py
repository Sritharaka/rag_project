import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import Pinecone as LangchainPinecone
import traceback
import torch

# ✅ Environment Variables
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_IfNGzzdUOOaVmiLcUCxJUgkjADOBYNOIE"
os.environ["PINECONE_API_KEY"] = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMg"

# ✅ Configuration
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"
MODEL_NAME = "tiiuae/falcon-7b-instruct"
MAX_NEW_TOKENS = 256

# ✅ Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Load Pinecone Vector Store
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

# ✅ Load Falcon-7B-Instruct Model (use float16 if supported)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # Automatically use GPU/CPU
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# ✅ Setup Pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id  # Avoid warning
)

# ✅ LangChain HuggingFacePipeline (correct import)
llm = HuggingFacePipeline(pipeline=generation_pipeline)

# ✅ Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # Faster and simpler for short context
)

# ✅ Flask Setup
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
    app.run(debug=True)
