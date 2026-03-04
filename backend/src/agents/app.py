import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agents.graph import create_lead_graph

logger = logging.getLogger(__name__)

langraph_api = FastAPI(
    title="Langgraph API",
    description="API for interacting with the LangGraph server and agents",
    version="1.0.0",
)

# Add CORS middleware
# It's recommended to load the allowed origins from an environment variable
# for better security and flexibility across different environments.

langraph_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Use the configured list of methods
    allow_headers=["*"],  # Now allow all headers, but can be restricted further
)

graph = create_lead_graph()

# Implement the following langraph API endpoints:
# /api/langgraph/threads/search
# /api/langgraph/threads/{thread_id}/history
# /api/langgraph/threads/{thread_id}/runs/stream
