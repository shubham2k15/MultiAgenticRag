# ============================================================
# app.py
# FastAPI wrapper around agent.py for Cloud Run deployment
# ============================================================

import warnings

from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning
)

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning
)


import os

from fastapi import FastAPI, HTTPException

from pydantic import BaseModel

from langchain_core.documents import Document


# Import the compiled graph builder from agent.py
from agent import build_rag_graph, GraphState


app = FastAPI(
    title="Agentic RAG",
    description="LangGraph + Groq + Vertex AI Vector Search",
    version="1.0.0",
)


# ============================================================
# CORS
# Required for React frontend to call Cloud Run backend
# ============================================================

from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "https://multiagenticrag-510714.web.app",          # your Firebase domain
        #"https://agentic-rag-495906.web.app",        # alternate Firebase domain
        "http://localhost:5173",                     # local dev
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# Build graph ONCE at startup
# Expensive - don't rebuild per request
# ============================================================

rag_app = build_rag_graph()


# ============================================================
# Request / Response models
# ============================================================

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    route: str
    answer: str


# ============================================================
# Health check
# ============================================================

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Agentic RAG"
    }


# ============================================================
# Main chat endpoint
# ============================================================

@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # ========================================================
    # Initial LangGraph state
    # ========================================================

    initial_state: GraphState = {
        "question": request.question,
        "route": "",
        "vector_docs": [],
        "web_results": [],
        "final_answer": "",
    }


    # ========================================================
    # Run the LangGraph
    # ========================================================

    result = rag_app.invoke(initial_state)


    # ========================================================
    # Return response
    # ========================================================

    return QueryResponse(
        question=request.question,
        route=result["route"],
        answer=result["final_answer"],
    )