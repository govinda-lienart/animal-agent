# vector_lookup_v6.py

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# Load environment variables (API key, etc.)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
INDEX_DIR = "faiss_indexes_v6"

# Strict prompt to reduce hallucinations
custom_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are an expert animal research assistant. Use only the information in the context below to answer the question.
If the answer cannot be found in the context, respond with:
"I couldn't find that information in the provided sources."

Context:
{context}

Question: {input}
Answer:"""
)

# Load all FAISS indexes
def load_all_indexes():
    all_stores = []
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    for folder in os.listdir(INDEX_DIR):
        path = os.path.join(INDEX_DIR, folder)
        if os.path.isdir(path):
            try:
                store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                all_stores.append(store)
            except Exception as e:
                print(f"Failed to load index {folder}: {e}")
    return all_stores

# Combine all indexes into one
all_vectorstores = load_all_indexes()
combined_vectorstore = all_vectorstores[0]
for store in all_vectorstores[1:]:
    combined_vectorstore.merge_from(store)

# Set up the QA chain using strict prompt
llm = OpenAI(openai_api_key=api_key)
combine_docs_chain = create_stuff_documents_chain(llm, custom_prompt)
retrieval_chain = create_retrieval_chain(combined_vectorstore.as_retriever(), combine_docs_chain)

# Function to handle user question
def ask_question(query: str) -> dict:
    response = retrieval_chain.invoke({"input": query})

    # Collect unique source files
    sources = []
    seen_files = set()
    for doc in response.get("context", []):
        file_name = doc.metadata.get("file_name", "Unknown file")
        if file_name not in seen_files:
            sources.append({"file_name": file_name})
            seen_files.add(file_name)

    return {
        "answer": response["answer"],
        "sources": sources
    }