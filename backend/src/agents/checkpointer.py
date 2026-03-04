import contextlib
import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

"""
langraph.json:
  
  "checkpointer": {
    "path": "src.agents.create_postgres_checkpointer"
  },
  "store":{
      "path": "src.agents.create_postgres_store"
  }
  
"""
@contextlib.asynccontextmanager
async def create_postgres_checkpointer():
   connection_kwargs = {
        "autocommit": True,
        "row_factory": "dict_row",
        "prepare_threshold": 0,
    }
   db_url = os.getenv("DATABASE_URL")
   async with AsyncConnectionPool( db_url, kwargs=connection_kwargs) as conn:
     checkpointer = AsyncPostgresSaver(conn)
     checkpointer.setup()
     yield checkpointer
    