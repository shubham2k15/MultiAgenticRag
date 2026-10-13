# ============================================================
# ingest.py
# Run this ONCE to:
# 1. Parse your PDF
# 2. Chunk the text
# 3. Create Vertex AI Vector Search Index + Endpoint
# 4. Ingest embeddings + chunks into Vector Search
# ============================================================

import os
import re
import warnings

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai.vectorstores import VectorSearchVectorStore

from google.cloud import aiplatform
from langchain_core._api.deprecation import LangChainDeprecationWarning


# Suppress deprecation warnings
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

INDEX_NAME = os.getenv("INDEX_DISPLAY_NAME")
ENDPOINT_NAME = os.getenv("ENDPOINT_DISPLAY_NAME")


# ============================================================
# PDF files
# ============================================================

PDF_PATHS = [
    r"C:\Users\SHUBHAM\Documents\MultiAgenticRag\Files\jack-and-the-beanstalk-joseph-jacobs-4940.pdf",

    # Add more PDFs if required
    # r"pdfs/second_file.pdf",
]


# ============================================================
# Step 1: Initialize GCP
# ============================================================

print("\nInitializing GCP...")

aiplatform.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=f"gs://{BUCKET}",
)

print("GCP initialized.")


# ============================================================
# Step 2: Load PDFs
# ============================================================

print("\nLoading PDF(s)...")

all_docs = []

for path in PDF_PATHS:

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
# Step 3: Chunk documents
# ============================================================

print("\nChunking documents...")

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
# Preview a chunk
# ============================================================

if chunks:

    print(
        f"\nPreview of chunk 1:\n"
        f"{chunks[0].page_content[:200]}..."
    )

else:

    raise ValueError(
        "No chunks were created. Check the PDF path and PDF content."
    )


# ============================================================
# Step 4: Set up Vertex AI Embeddings
# ============================================================

print(
    "\nSetting up Vertex AI Embeddings "
    "(text-embedding-005)..."
)

embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=PROJECT_ID,
    location=REGION,
)

print("Embeddings model ready.")


# ============================================================
# Step 5: Create Vector Search Index
# ============================================================

print(
    "\nCreating Vertex AI Vector Search Index..."
)

print(
    "This can take several minutes. Please wait..."
)

my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(

    display_name=INDEX_NAME,

    # text-embedding-005 produces 768-dimensional vectors
    dimensions=768,

    approximate_neighbors_count=10,

    leaf_node_embedding_count=500,

    leaf_nodes_to_search_percent=7,

    index_update_method="STREAM_UPDATE",

    distance_measure_type="DOT_PRODUCT_DISTANCE",
)

print("Index created!")

print(
    f"Index Resource Name: "
    f"{my_index.resource_name}"
)


# ============================================================
# Step 6: Create Vector Search Index Endpoint
# ============================================================

print(
    "\nCreating Vector Search Index Endpoint..."
)

my_index_endpoint = (
    aiplatform.MatchingEngineIndexEndpoint.create(

        display_name=ENDPOINT_NAME,

        public_endpoint_enabled=True,
    )
)

print("Endpoint created!")

print(
    f"Endpoint Resource Name: "
    f"{my_index_endpoint.resource_name}"
)


# ============================================================
# Step 7: Deploy Index to Endpoint
# ============================================================

print(
    "\nDeploying Index to Endpoint..."
)

print(
    "This can take several minutes. Please wait..."
)

my_index_endpoint.deploy_index(

    index=my_index,

    deployed_index_id=(
        "multiagenticrag_510714_deployed_index"
    ),
)

print(
    "Index successfully deployed to Endpoint!"
)


# ============================================================
# Step 8: Create Vector Store
# ============================================================

print(
    "\nCreating Vertex AI Vector Search Vector Store..."
)

vector_store = VectorSearchVectorStore.from_components(

    project_id=PROJECT_ID,

    region=REGION,

    gcs_bucket_name=BUCKET,

    index_id=my_index.name,

    endpoint_id=my_index_endpoint.name,

    embedding=embeddings,

    stream_update=True,
)


# ============================================================
# Step 9: Ingest documents
# ============================================================

print(
    "\nIngesting chunks into Vector Search..."
)

print(
    "Generating embeddings and indexing chunks..."
)

vector_store.add_documents(chunks)

print(
    f"All {len(chunks)} chunks "
    f"ingested successfully!"
)


# ============================================================
# Step 10: Save Vector Search IDs
# ============================================================

with open(
    "vector_store_ids.txt",
    "w"
) as f:

    f.write(
        f"INDEX_ID={my_index.name}\n"
    )

    f.write(
        f"ENDPOINT_ID={my_index_endpoint.name}\n"
    )


# ============================================================
# Done
# ============================================================

print("\n" + "=" * 60)

print("INGESTION COMPLETE!")

print("=" * 60)

print(
    "Index ID saved → vector_store_ids.txt"
)

print(
    "Endpoint ID saved → vector_store_ids.txt"
)

print(
    "\nNow your agent can use these IDs "
    "to connect to Vector Search."
)

print("=" * 60)