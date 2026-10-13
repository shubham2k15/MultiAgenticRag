# ============================================================

# agent.py

#

# Multi-Agent RAG using:

#   1. Groq LLM

#   2. Vertex AI Embeddings

#   3. Vertex AI Vector Search

#   4. Tavily Web Search

#   5. LangGraph

#

# Overall flow:

#

#                    User Question

#                         |

#                         v

#                  +--------------+

#                  | Orchestrator |

#                  +------+-------+

#                         |

#              +----------+----------+

#              |          |          |

#              v          v          v

#          Vector DB   Web Search   Direct

#              |          |          |

#              +-----+----+          |

#                    |               |

#                    v               v

#                 Generate          Answer

#                    |

#                    v

#                   END

#

# The orchestrator can choose:

#   - vector_db

#   - web_search

#   - both

#   - direct

# ============================================================





# ============================================================

# 1. IMPORTS

# ============================================================



import os

from typing import TypedDict



# Load variables from .env

from dotenv import load_dotenv



# ------------------------------------------------------------

# Groq LLM

# ------------------------------------------------------------

# Used for:

#   - Deciding which route to use

#   - Direct answers

#   - Final answer generation

# ------------------------------------------------------------

from langchain_groq import ChatGroq



# ------------------------------------------------------------

# Vertex AI Embeddings

# ------------------------------------------------------------

# Converts text into numerical vectors.

#

# Example:

#

# "Who is Rowan Iyer?"

#          |

#          v

# Embedding model

#          |

#          v

# [0.12, -0.45, 0.78, ...]

# ------------------------------------------------------------

from langchain_google_vertexai import VertexAIEmbeddings



# ------------------------------------------------------------

# LangChain interface for Vertex AI Vector Search

# ------------------------------------------------------------

from langchain_google_vertexai.vectorstores import (

    VectorSearchVectorStore

)



# ------------------------------------------------------------

# Tavily Web Search

# ------------------------------------------------------------

from langchain_community.tools.tavily_search import (

    TavilySearchResults

)



# ------------------------------------------------------------

# Google Cloud / Vertex AI SDK

# ------------------------------------------------------------

from google.cloud import aiplatform



# ------------------------------------------------------------

# LangChain message types

# ------------------------------------------------------------

from langchain_core.messages import (

    SystemMessage,

    HumanMessage

)



# ------------------------------------------------------------

# LangGraph

# ------------------------------------------------------------

# StateGraph → creates our graph

# START      → graph starting point

# END        → graph ending point

# ------------------------------------------------------------

from langgraph.graph import (

    StateGraph,

    START,

    END

)





# ============================================================

# 2. LOAD ENVIRONMENT VARIABLES

# ============================================================



# Read values from .env

load_dotenv()





# ------------------------------------------------------------

# Google Cloud configuration

# ------------------------------------------------------------



# Google Cloud project ID

PROJECT_ID = os.getenv("GCP_PROJECT_ID")



# Google Cloud region

# If GCP_REGION is not present, use us-central1

REGION = os.getenv(

    "GCP_REGION",

    "us-central1"

)



# GCS bucket used by the Vector Search setup

BUCKET = os.getenv("GCS_BUCKET")





# ------------------------------------------------------------

# API keys

# ------------------------------------------------------------



# Groq API key

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Tavily API key

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")





# ============================================================

# 3. VECTOR SEARCH IDS

# ============================================================



# These values will be loaded from:

#

# vector_store_ids.txt

#

# This file is created by ingest.py.

# ------------------------------------------------------------



INDEX_ID = None

ENDPOINT_ID = None





# ============================================================

# 4. READ VECTOR SEARCH IDs

# ============================================================



# ingest.py creates this file after creating:

#

#   1. Vector Search Index

#   2. Vector Search Endpoint

#

# The IDs are stored in this file so that agent.py

# doesn't need to create the index again.

# ------------------------------------------------------------



IDS_FILE = "vector_store_ids.txt"

print("\n===== DEBUG: VECTOR STORE FILE =====")

print("Current directory:", os.getcwd())

print("IDS_FILE:", IDS_FILE)

print("Absolute path:", os.path.abspath(IDS_FILE))

print("File exists:", os.path.exists(IDS_FILE))



# Check whether the file exists

if os.path.exists(IDS_FILE):



    # Open the file

    with open(

        IDS_FILE,

        "r",

        encoding="utf-8-sig"

    ) as f:



        # Read the file line by line

        for line in f:



            # Example line:

            #

            # INDEX_ID=projects/...

            #

            # partition("=") separates the key and value.

            print("Raw line:", repr(line))

            key, _, value = line.strip().partition("=")

            print("Key:", repr(key))

            print("Value:", repr(value))



            # Read INDEX_ID

            if key == "INDEX_ID":

                INDEX_ID = value



            # Read ENDPOINT_ID

            elif key == "ENDPOINT_ID":

                ENDPOINT_ID = value



    print("Final INDEX_ID:", repr(INDEX_ID))

    print("Final ENDPOINT_ID:", repr(ENDPOINT_ID))

    print("=================================\n")

# ============================================================

# 5. LAZY-LOADED AI COMPONENTS

# ============================================================

#

# We don't initialize these immediately.

#

# They will be created when the first request arrives.

#

# This is called lazy initialization.

#

# It is useful when deploying the application to Cloud Run

# because it avoids doing expensive initialization immediately

# when the container starts.

# ============================================================



llm = None



retriever = None



web_search = None





# ============================================================

# 6. INITIALIZE AI COMPONENTS

# ============================================================



def initialize_components():



    """

    Initialize all AI components used by the application.



    Components:

        1. Google Cloud / Vertex AI

        2. Groq LLM

        3. Vertex AI Embeddings

        4. Vertex AI Vector Search

        5. Tavily Web Search



    The components are initialized only once.

    """



    # We are modifying the global variables defined above.

    global llm, retriever, web_search





    # --------------------------------------------------------

    # Don't initialize again if everything already exists.

    # --------------------------------------------------------



    if (

        llm is not None

        and retriever is not None

        and web_search is not None

    ):

        return





    # ========================================================

    # Validate configuration

    # ========================================================



    # Google Cloud project

    if not PROJECT_ID:

        raise ValueError(

            "GCP_PROJECT_ID is missing from .env"

        )





    # Google Cloud region

    if not REGION:

        raise ValueError(

            "GCP_REGION is missing from .env"

        )





    # GCS bucket

    if not BUCKET:

        raise ValueError(

            "GCS_BUCKET is missing from .env"

        )





    # Vector Search IDs

    #

    # These should have been created by ingest.py.

    #

    if not INDEX_ID or not ENDPOINT_ID:



        raise ValueError(

            "INDEX_ID or ENDPOINT_ID is missing. "

            "Run ingest.py first so "

            "vector_store_ids.txt is created."

        )





    # Groq API key

    if not GROQ_API_KEY:



        raise ValueError(

            "GROQ_API_KEY is missing from .env"

        )





    # Tavily API key

    if not TAVILY_API_KEY:



        raise ValueError(

            "TAVILY_API_KEY is missing from .env"

        )





    # ========================================================

    # 6.1 Initialize Vertex AI

    # ========================================================



    aiplatform.init(

        project=PROJECT_ID,

        location=REGION,

        staging_bucket=f"gs://{BUCKET}",

    )





    # ========================================================

    # 6.2 Create Groq LLM

    # ========================================================

    #

    # This LLM is used for:

    #

    #   1. Routing

    #   2. Direct answers

    #   3. Final answer generation

    #

    # temperature=0 makes the model more deterministic.

    # ========================================================



    llm = ChatGroq(

    model="openai/gpt-oss-120b",

    temperature=0,

    api_key=GROQ_API_KEY,

        )





    # ========================================================

    # 6.3 Create Vertex AI Embedding Model

    # ========================================================

    #

    # This model converts text into vectors.

    #

    # IMPORTANT:

    # The same embedding model must be used during:

    #

    #   ingest.py

    #

    # and

    #

    #   agent.py

    #

    # because the query and stored document vectors need

    # to exist in the same embedding space.

    # ========================================================



    embeddings = VertexAIEmbeddings(

        model_name="text-embedding-005",

        project=PROJECT_ID,

        location=REGION,

    )





    # ========================================================

    # 6.4 Connect to Vertex AI Vector Search

    # ========================================================

    #

    # ingest.py already created the:

    #

    #   Index

    #   Endpoint

    #

    # Here we connect to those existing resources.

    # ========================================================



    vector_store = VectorSearchVectorStore.from_components(



        project_id=PROJECT_ID,



        region=REGION,



        gcs_bucket_name=BUCKET,



        index_id=INDEX_ID,



        endpoint_id=ENDPOINT_ID,



        embedding=embeddings,



        stream_update=True,

    )





    # ========================================================

    # 6.5 Create Retriever

    # ========================================================

    #

    # k=4 means:

    #

    # Return the top 4 relevant chunks from Vector Search.

    # ========================================================



    retriever = vector_store.as_retriever(

        search_kwargs={

            "k": 4

        }

    )





    # ========================================================

    # 6.6 Create Tavily Web Search

    # ========================================================

    #

    # max_results=3 means Tavily can return up to

    # 3 search results.

    # ========================================================



    web_search = TavilySearchResults(

        max_results=3,

        tavily_api_key=TAVILY_API_KEY,

    )





# ============================================================

# 7. LANGGRAPH STATE

# ============================================================

#

# LangGraph nodes communicate using a shared state.

#

# Think of GraphState as a common container:

#

#     question

#     route

#     vector_docs

#     web_results

#     final_answer

#

# Each node can read information from the state and

# return updated information.

# ============================================================



class GraphState(TypedDict, total=False):



    # Original user question

    question: str



    # Route selected by the orchestrator

    #

    # Possible values:

    #   vector_db

    #   web_search

    #   both

    #   direct

    route: str



    # Documents retrieved from Vector Search

    vector_docs: list



    # Results returned from Tavily

    web_results: list



    # Final answer generated by the LLM

    final_answer: str





# ============================================================

# 8. ORCHESTRATOR NODE

# ============================================================

#

# The orchestrator is the decision-making part of the system.

#

# It receives:

#

#     User question

#

# and decides:

#

#     vector_db

#     web_search

#     both

#     direct

# ============================================================



def orchestrator(
    state: GraphState
) -> GraphState:

    # Make sure all AI components are initialized.
    initialize_components()

    # Get the user's question from the shared state.
    question = state["question"]

    # --------------------------------------------------------
    # Prompt used by the routing LLM
    # --------------------------------------------------------

    system_prompt = """
You are a routing agent for a multi-agent RAG system.

Your job is ONLY to decide which information source is required
to answer the user's question.

Return ONLY ONE of these four values:

vector_db
web_search
both
direct


ROUTING RULES
=============

1. vector_db

Use vector_db when the answer should come from the private
documents stored in the Vector Search database.


2. web_search

Use web_search when the answer requires information from the
internet, especially current, latest, real-time, or external
information that is not expected to be in the private documents.


3. both

Use both when the question requires BOTH:

A. Information from the private documents
AND
B. Information from the web.

IMPORTANT:

If the user explicitly asks to use the provided/private document
AND also asks to search the web, the route MUST be both.

Examples:

- From the provided document, identify the author of Jack and
  the Beanstalk. Use web search to find the author's nationality.

- According to the document, who wrote Jack and the Beanstalk,
  and what other books did the author write?

- Summarize the story from the document and tell me when it
  was first published.


4. direct

Use direct ONLY when neither the private documents nor web
search is required.

Examples:

- Hi
- Hello
- What can you do?
- Thanks


DECISION PRIORITY
=================

1. If BOTH private-document information and web information
   are required -> both

2. If ONLY private-document information is required -> vector_db

3. If ONLY web information is required -> web_search

4. If NEITHER is required -> direct


Do not answer the user's question.

Do not explain your decision.

Return ONLY the routing value.
"""

    # --------------------------------------------------------
    # Ask the LLM to select the route
    # --------------------------------------------------------

    response = llm.invoke(
        [
            SystemMessage(
                content=system_prompt
            ),

            HumanMessage(
                content=question
            ),
        ]
    )

    # Get the LLM response.
    route = response.content.strip().lower()

    # --------------------------------------------------------
    # Clean accidental formatting
    # --------------------------------------------------------

    route = (
        route
        .replace("`", "")
        .replace(".", "")
        .strip()
    )

    # ========================================================
    # Normalize the route
    # ========================================================

    # Handle an exact "both" response first.
    if route == "both":
        route = "both"

    # Handle responses containing both terms.
    elif "vector" in route and "web" in route:
        route = "both"

    # Handle Vector Search responses.
    elif "vector" in route:
        route = "vector_db"

    # Handle web/search responses.
    elif "web" in route or "search" in route:
        route = "web_search"

    # Handle direct responses.
    elif "direct" in route:
        route = "direct"

    # Unexpected response → safe fallback.
    else:
        route = "direct"

    # Debug information.
    print(
        f"\n[ROUTER] Question: {question}"
    )

    print(
        f"[ROUTER] LLM response: {response.content!r}"
    )

    print(
        f"[ROUTER] Final route: {route}"
    )

    # Return the selected route back into GraphState.
    return {
        "route": route
    }


# ============================================================

# 9. DIRECT LLM NODE

# ============================================================

#

# Used when the question doesn't require:

#

#   - Vector Search

#   - Web Search

#

# Example:

#

#   User: Hi

#

# The orchestrator chooses:

#

#   direct

#

# Then this node answers the user directly.

# ============================================================



def direct_llm(

    state: GraphState

) -> GraphState:



    # Make sure LLM is initialized.

    initialize_components()





    # Get user question.

    question = state["question"]





    # Prompt for normal conversation.

    system_prompt = """

You are a friendly and helpful AI assistant.



Answer the user's query normally and conversationally.



Do not use external tools unless they are explicitly provided.

"""





    # Ask the LLM for the answer.

    response = llm.invoke(

        [

            SystemMessage(

                content=system_prompt

            ),



            HumanMessage(

                content=question

            ),

        ]

    )





    # Store answer in GraphState.

    return {

        "final_answer": response.content

    }





# ============================================================

# 10. VECTOR DATABASE RETRIEVAL NODE

# ============================================================

#

# Used when the orchestrator selects:

#

#     vector_db

#

# or

#

#     both

#

# It searches the private documents stored in

# Vertex AI Vector Search.

# ============================================================



def retrieve_from_vector_db(

    state: GraphState

) -> GraphState:



    # Initialize components if required.

    initialize_components()





    # Get user question.

    question = state["question"]





    # --------------------------------------------------------

    # Search Vector Search

    # --------------------------------------------------------

    #

    # The retriever was configured with:

    #

    #     k=4

    #

    # So we retrieve the top 4 relevant chunks.

    # --------------------------------------------------------



    docs = retriever.invoke(question)





    # List that will contain simplified document data.

    vector_docs = []





    # --------------------------------------------------------

    # Convert LangChain Documents into dictionaries

    # --------------------------------------------------------



    for doc in docs:



        vector_docs.append(

            {

                # Actual text retrieved from the document

                "content": doc.page_content,



                # Complete metadata

                "metadata": doc.metadata,



                # Document source

                "source": doc.metadata.get(

                    "source",

                    ""

                ),



                # Page number

                "page": doc.metadata.get(

                    "page",

                    ""

                ),

            }

        )





    # Return retrieved documents to GraphState.

    return {

        "vector_docs": vector_docs

    }





# ============================================================

# 11. WEB SEARCH NODE

# ============================================================

#

# Used when the orchestrator selects:

#

#     web_search

#

# or when:

#

#     both

#

# is selected and Vector Search has already run.

# ============================================================



def search_web(

    state: GraphState

) -> GraphState:



    # Make sure components are initialized.

    initialize_components()





    # Get user question.

    question = state["question"]





    # --------------------------------------------------------

    # Search the web using Tavily

    # --------------------------------------------------------



    results = web_search.invoke(question)





    # List to store normalized web results.

    web_results = []





    # --------------------------------------------------------

    # Process each search result

    # --------------------------------------------------------



    for result in results:



        # Tavily normally returns dictionaries.

        if isinstance(result, dict):



            web_results.append(

                {

                    # Search result title

                    "title": result.get(

                        "title",

                        ""

                    ),



                    # Search result content

                    #

                    # Some result formats may use "snippet"

                    # instead of "content", so we provide

                    # snippet as a fallback.

                    "content": result.get(

                        "content",

                        result.get(

                            "snippet",

                            ""

                        )

                    ),



                    # Web page URL

                    "url": result.get(

                        "url",

                        ""

                    ),

                }

            )



        # Fallback if result isn't a dictionary.

        else:



            web_results.append(

                {

                    "title": "",

                    "content": str(result),

                    "url": "",

                }

            )





    # Return web results to GraphState.

    return {

        "web_results": web_results

    }





# ============================================================

# 12. FINAL GENERATION NODE

# ============================================================

#

# This node creates the final answer.

#

# It can receive:

#

#     Private document context

#

#     Web search context

#

#     Or both

#

# Then the LLM uses that context to answer the question.

# ============================================================



def generate(

    state: GraphState

) -> GraphState:



    # Initialize components if required.

    initialize_components()





    # Get user's question.

    question = state["question"]





    # This list will contain all retrieved context.

    context_parts = []





    # ========================================================

    # 12.1 PRIVATE DOCUMENT CONTEXT

    # ========================================================



    # Get Vector Search results.

    #

    # If none exist, return an empty list.

    vector_docs = state.get(

        "vector_docs",

        []

    )





    # If Vector Search returned documents,

    # add them to the context.

    if vector_docs:



        context_parts.append(

            "PRIVATE DOCUMENT CONTEXT:"

        )





        # Process each retrieved chunk.

        for i, doc in enumerate(

            vector_docs,

            start=1

        ):



            context_parts.append(

                f"""

Document Chunk {i}



Source: {doc.get("source", "")}



Page: {doc.get("page", "")}



Content:



{doc.get("content", "")}

"""

            )





    # ========================================================

    # 12.2 WEB SEARCH CONTEXT

    # ========================================================



    # Get Tavily results.

    web_results = state.get(

        "web_results",

        []

    )





    # If web results exist,

    # add them to the context.

    if web_results:



        context_parts.append(

            "WEB SEARCH CONTEXT:"

        )





        # Process each web result.

        for i, result in enumerate(

            web_results,

            start=1

        ):



            context_parts.append(

                f"""

Web Result {i}



Title: {result.get("title", "")}



URL: {result.get("url", "")}



Content:



{result.get("content", "")}

"""

            )





    # --------------------------------------------------------

    # Combine all context into one string.

    # --------------------------------------------------------



    context = "\n".join(

        context_parts

    )





    # ========================================================

    # 12.3 FINAL LLM PROMPT

    # ========================================================



    system_prompt = f"""

You are a helpful and accurate AI assistant.



Answer the user's question using the context provided below.



Rules:



- Use the retrieved context when it is relevant.

- Do not invent facts that are not supported by the context.

- If the context does not contain enough information, clearly say so.

- When web results are provided, use them for current information.

- Give a clear, concise and useful answer.



CONTEXT:



{context}

"""





    # --------------------------------------------------------

    # Ask the LLM to generate final answer.

    # --------------------------------------------------------



    response = llm.invoke(

        [

            SystemMessage(

                content=system_prompt

            ),



            HumanMessage(

                content=question

            ),

        ]

    )





    # Store final answer in GraphState.

    return {

        "final_answer": response.content

    }





# ============================================================

# 13. ROUTING AFTER ORCHESTRATOR

# ============================================================

#

# This function tells LangGraph which node to execute

# after the orchestrator.

#

# Possible routes:

#

#     vector_db

#     web_search

#     both

#     direct

# ============================================================



def route_after_orchestrator(

    state: GraphState

):



    # Get route from state.

    route = state.get(

        "route",

        "direct"

    )





    # --------------------------------------------------------

    # vector_db

    # --------------------------------------------------------



    if route == "vector_db":



        return "retrieve_from_vector_db"





    # --------------------------------------------------------

    # web_search

    # --------------------------------------------------------



    if route == "web_search":



        return "search_web"





    # --------------------------------------------------------

    # both

    #

    # Vector Search runs first.

    # After Vector Search, another routing function

    # will send the flow to Web Search.

    # --------------------------------------------------------



    if route == "both":



        return "retrieve_from_vector_db"





    # --------------------------------------------------------

    # direct

    #

    # Also acts as the fallback route.

    # --------------------------------------------------------



    return "direct_llm"





# ============================================================

# 14. ROUTING AFTER VECTOR DATABASE

# ============================================================

#

# This function is used after Vector Search.

#

# There are two possibilities:

#

#     both

#        ↓

#     Web Search

#

#     vector_db

#        ↓

#     Generate

# ============================================================



def route_after_vector_db(

    state: GraphState

):



    # Get route from state.

    route = state.get(

        "route",

        "vector_db"

    )





    # If original route was "both",

    # perform web search as well.

    if route == "both":



        return "search_web"





    # Otherwise go directly to final generation.

    return "generate"





# ============================================================

# 15. BUILD LANGGRAPH

# ============================================================



def build_rag_graph():



    """

    Create and compile the complete LangGraph.



    Graph structure:



                    START

                      |

                      v

                Orchestrator

                      |

          +-----------+-----------+

          |           |           |

          v           v           v

       Vector DB   Web Search   Direct LLM

          |           |           |

          |           |           |

          +-----+-----+           |

                |                 |

                v                 |

             Generate             |

                |                 |

                v                 |

               END <--------------+

    """



    # Initialize AI components.

    initialize_components()





    # --------------------------------------------------------

    # Create StateGraph using our GraphState

    # --------------------------------------------------------



    graph = StateGraph(

        GraphState

    )





    # ========================================================

    # 15.1 Add nodes

    # ========================================================



    # Orchestrator

    graph.add_node(

        "orchestrator",

        orchestrator

    )





    # Direct LLM

    graph.add_node(

        "direct_llm",

        direct_llm

    )





    # Vector Search retrieval

    graph.add_node(

        "retrieve_from_vector_db",

        retrieve_from_vector_db

    )





    # Tavily Web Search

    graph.add_node(

        "search_web",

        search_web

    )





    # Final answer generation

    graph.add_node(

        "generate",

        generate

    )





    # ========================================================

    # 15.2 START → Orchestrator

    # ========================================================



    graph.add_edge(

        START,

        "orchestrator"

    )





    # ========================================================

    # 15.3 Orchestrator → next node

    # ========================================================

    #

    # route_after_orchestrator() determines the destination.

    # ========================================================



    graph.add_conditional_edges(



        "orchestrator",



        route_after_orchestrator,



        {

            "retrieve_from_vector_db":

                "retrieve_from_vector_db",



            "search_web":

                "search_web",



            "direct_llm":

                "direct_llm",

        },

    )





    # ========================================================

    # 15.4 Vector DB → Web Search / Generate

    # ========================================================

    #

    # If route = both:

    #

    #     Vector DB → Web Search

    #

    # Otherwise:

    #

    #     Vector DB → Generate

    # ========================================================



    graph.add_conditional_edges(



        "retrieve_from_vector_db",



        route_after_vector_db,



        {

            "search_web":

                "search_web",



            "generate":

                "generate",

        },

    )





    # ========================================================

    # 15.5 Web Search → Generate

    # ========================================================



    graph.add_edge(

        "search_web",

        "generate"

    )





    # ========================================================

    # 15.6 Generate → END

    # ========================================================



    graph.add_edge(

        "generate",

        END

    )





    # ========================================================

    # 15.7 Direct LLM → END

    # ========================================================



    graph.add_edge(

        "direct_llm",

        END

    )





    # --------------------------------------------------------

    # Compile the graph.

    #

    # After compilation, the graph can be executed using:

    #

    #     graph.invoke(...)

    # --------------------------------------------------------



    return graph.compile()





# ============================================================

# 16. LOCAL TESTING

# ============================================================

#

# This section runs ONLY when you execute:

#

#     python agent.py

#

# It does NOT run when app.py imports agent.py.

# ============================================================



if __name__ == "__main__":



    # Build the complete RAG graph.

    graph = build_rag_graph()





    # Display header.

    print(

        "=" * 70

    )



    print(

        "MULTI-AGENT RAG TEST"

    )



    print(

        "=" * 70

    )





    # --------------------------------------------------------

    # Keep asking questions until user types "exit"

    # --------------------------------------------------------



    while True:



        question = input(

            "\nAsk a question (or type 'exit'): "

        ).strip()





        # Exit the program.

        if question.lower() == "exit":



            break





        # Ignore empty questions.

        if not question:



            continue





        # ====================================================

        # Create initial LangGraph state

        # ====================================================



        initial_state: GraphState = {



            # User's question

            "question": question,



            # Initially no route has been selected.

            # The orchestrator will decide it.

            "route": "",



            # Initially there are no retrieved documents.

            "vector_docs": [],



            # Initially there are no web results.

            "web_results": [],



            # Initially there is no final answer.

            "final_answer": "",

        }





        # ====================================================

        # Execute the LangGraph

        # ====================================================



        result = graph.invoke(

            initial_state

        )





        # ====================================================

        # Display the result

        # ====================================================



        print(

            "\nRoute:",

            result.get("route")

        )



        print(

            "\nAnswer:"

        )



        print(

            result.get("final_answer")

        )



        print(

            "\n" + "-" * 70

        )