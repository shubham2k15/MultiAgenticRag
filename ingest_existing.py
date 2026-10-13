# ============================================================
# ingest_existing.py
#
# Run this to:
# 1. Parse your PDF
# 2. Clean and chunk the text
# 3. Create embeddings
# 4. Add the chunks to your EXISTING
#    Vertex AI Vector Search Index
#
# IMPORTANT:
# This script DOES NOT create a new Index.
# This script DOES NOT create a new Endpoint.
# ============================================================

import os
import re
import warnings

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

from langchain_core._api.deprecation import LangChainDeprecationWarning


# ============================================================
# Suppress warnings
# ============================================================

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)


# ============================================================
# Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_REGION")
BUCKET = os.getenv("GCS_BUCKET")


# ============================================================
# EXISTING Vertex AI Vector Search resources
# ============================================================

# We are REUSING the resources you already created.
#
# Existing Index:
# 1142108907259297792
#
# Existing Endpoint:
# 7865754452504870912

INDEX_ID = "1142108907259297792"

ENDPOINT_ID = "7865754452504870912"


# ============================================================
# PDF files
# ============================================================

PDF_PATHS = [
    r"C:\Users\SHUBHAM\Documents\MultiAgenticRag\Files\jack-and-the-beanstalk-joseph-jacobs-4940.pdf",
]


# ============================================================
# Validate configuration
# ============================================================

if not PROJECT_ID:
    raise ValueError("GCP_PROJECT_ID is missing from .env")

if not REGION:
    raise ValueError("GCP_REGION is missing from .env")

if not BUCKET:
    raise ValueError("GCS_BUCKET is missing from .env")


# ============================================================
# Step 1: Load PDFs
# ============================================================

print("\n" + "=" * 60)
print("STEP 1: Loading PDF")
print("=" * 60)

all_docs = []

for path in PDF_PATHS:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found:\n{path}"
        )

    loader = PyPDFLoader(path)

    docs = loader.load()

    # --------------------------------------------------------
    # Clean PDF text BEFORE chunking
    # --------------------------------------------------------

    for doc in docs:

        text = doc.page_content

        # Merge broken lines inside paragraphs
        text = re.sub(
            r"(?<!\n)\n(?!\n)",
            " ",
            text
        )

        # Keep real paragraph breaks
        text = re.sub(
            r"\n{2,}",
            "\n\n",
            text
        )

        # Remove extra spaces/tabs
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )

        # Fix broken hyphenated words
        text = re.sub(
            r"-\s+",
            "",
            text
        )

        doc.page_content = text.strip()

    all_docs.extend(docs)

    print(
        f"Loaded: {path} → {len(docs)} pages"
    )


print(
    f"\nTotal pages loaded: {len(all_docs)}"
)


# ============================================================
# Step 2: Chunk documents
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Creating chunks")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",
        ". ",
        "? ",
        "! ",
        " ",
    ],
)

chunks = splitter.split_documents(all_docs)

print(
    f"Total chunks created: {len(chunks)}"
)


# ============================================================
# Preview chunks
# ============================================================

if not chunks:

    raise ValueError(
        "No chunks were created. "
        "Check the PDF path and PDF content."
    )


print("\nFirst chunk:")
print("-" * 60)
print(chunks[0].page_content)
print("-" * 60)


# ============================================================
# Step 3: Create Vertex AI Embeddings
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Setting up embeddings")
print("=" * 60)

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=PROJECT_ID,
    location=REGION,
)

print(
    "Embedding model: text-embedding-005"
)

print("Embeddings model ready.")


# ============================================================
# Step 4: Connect to EXISTING Vector Search
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Connecting to existing Vector Search")
print("=" * 60)

print(
    f"Index ID:    {INDEX_ID}"
)

print(
    f"Endpoint ID: {ENDPOINT_ID}"
)

print(
    f"Bucket:      {BUCKET}"
)


vector_store = VectorSearchVectorStore.from_components(

    project_id=PROJECT_ID,

    region=REGION,

    gcs_bucket_name=BUCKET,

    index_id=INDEX_ID,

    endpoint_id=ENDPOINT_ID,

    embedding=embeddings,

    stream_update=True,
)


print(
    "Connected to existing Vector Search."
)


# ============================================================
# Step 5: Ingest documents
# ============================================================

print("\n" + "=" * 60)
print("STEP 5: Ingesting chunks")
print("=" * 60)

print(
    "Generating embeddings and indexing chunks..."
)

print(
    "Please wait..."
)


vector_store.add_documents(chunks)


print(
    f"\nSuccessfully ingested {len(chunks)} chunks!"
)


# ============================================================
# Done
# ============================================================

print("\n" + "=" * 60)
print("INGESTION COMPLETE!")
print("=" * 60)

print(
    "\nYour existing Vector Search resources were used."
)

print(
    f"Index ID    : {INDEX_ID}"
)

print(
    f"Endpoint ID : {ENDPOINT_ID}"
)

print(
    "\nYou can now test your agent."
)

print("=" * 60)