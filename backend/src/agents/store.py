import contextlib
import os

from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool


@contextlib.asynccontextmanager
async def create_postgres_store():
   connection_kwargs = {
        "autocommit": True,
        "row_factory": "dict_row",
        "prepare_threshold": 0,
    }
   db_url = os.getenv("DATABASE_URL")
   async with AsyncConnectionPool( db_url, kwargs=connection_kwargs) as conn:
     store = AsyncPostgresStore(conn)
     store.setup()
     yield store