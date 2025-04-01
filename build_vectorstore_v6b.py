import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Paths
PDF_FOLDER = "/Users/govinda-dashugolienart/Library/CloudStorage/GoogleDrive-govinda.lienart@three-monkeys.org/My Drive/TMWC - Govinda /TMWC - Govinda /Data Science/Environments/Pycharm/Vectorstor-in-Memory/pdfs"
INDEX_DIR = "faiss_indexes_v6"
INDEX_STATE_FILE = "processed_files_v6.json"

os.makedirs(INDEX_DIR, exist_ok=True)

# Load already processed files
def load_index_state():
    if os.path.exists(INDEX_STATE_FILE):
        with open(INDEX_STATE_FILE, "r") as f:
            return json.load(f)
    return []

# Save updated processed list
def save_index_state(processed_files):
    with open(INDEX_STATE_FILE, "w") as f:
        json.dump(processed_files, f)

# Extract title from PDF (optional stub – you can improve it)
def extract_title(documents, fallback_title):
    for page in documents:
        text = page.page_content.strip().split('\n')[0]
        if text:
            return text.strip()
    return fallback_title.replace(".pdf", "")

# Process each PDF file
def process_pdf(file_path, file_name):
    print(f"📄 Processing: {file_name}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    title = extract_title(documents, file_name)

    # Add filename and title to metadata
    for doc in documents:
        doc.metadata["file_name"] = file_name
        doc.metadata["title"] = title

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = splitter.split_documents(documents)

    if not docs:
        print(f"⚠️ No text found in {file_name}. Skipping.")
        return False

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    index_path = os.path.join(INDEX_DIR, file_name.replace(".pdf", ""))
    vectorstore.save_local(index_path)
    return True

# Main run
processed_files = load_index_state()
new_files = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf") and file not in processed_files:
        full_path = os.path.join(PDF_FOLDER, file)
        if process_pdf(full_path, file):
            new_files.append(file)

# Save updated state
if new_files:
    processed_files += new_files
    save_index_state(processed_files)
    print("✅ Finished processing new PDFs.")
else:
    print("🟡 No new PDFs found.")
