rag_project/
├── frontend/ # React-based chatbot UI
├── backend/
│ ├── mistaralai.py # Backend API using HuggingFace model (Mistral)
│ ├── openai_test.py # Alternate RAG pipeline using OpenAI API
│ └── rag_tester.py # Python script to test RAG via command line
├── notebooks/
│ └── data_cleaning.ipynb # Notebook to clean data and upload to Pinecone

## 🔧 Prerequisites

- Node.js & npm
- Python 3.10+
- Pinecone API Key
- HuggingFace API Token and/or OpenAI API Key

---

##  Frontend: Chatbot UI

```bash
cd frontend
npm install
npm start