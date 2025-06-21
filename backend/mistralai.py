import os
import traceback
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    pipeline,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import Pinecone as LangchainPinecone

# Set Environment Variables (REPLACE with your actual keys)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_4DqA35_BYaWgQCoVUkGTNDRYenf3NQzwzZX6C685nC2fwMj5qXgnpMXmUcH1eVXRjVfMgw"

# Config
PINECONE_INDEX_NAME = "quickstart-new-test-csv-html-merged-final"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
MAX_NEW_TOKENS = 512

# Step 1: Load sentence transformer for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 2: Load Pinecone vector store
vectorstore = LangchainPinecone.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

# Step 3: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Step 4: Conditionally load model with bitsandbytes quantization if CUDA available, else load normally
if torch.cuda.is_available():
    print("CUDA detected: Loading quantized 4-bit model with bitsandbytes")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    print("CUDA NOT detected: Loading full precision model on CPU")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": "cpu"},
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

# Step 5: Create generation pipeline
generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

# Step 6: Wrap with LangChain interface
llm = HuggingFacePipeline(pipeline=generation_pipeline)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 7: RAG QA chain setup
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False
)

# Step 8: Flask app
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
