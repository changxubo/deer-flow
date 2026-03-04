import os
from redis import Redis
from langgraph.cache.redis import RedisCache

def create_redis_cache():
   prefix = "langgraph:cache:"
   redis_client =  Redis(
        host=os.getenv("REDIS_HOST","localhost"),  # or your Redis Enterprise endpoint
        port=int(os.getenv("REDIS_PORT", 6379)),  # or 10000 for Azure Managed Redis / Azure Enterprise with TLS
        password=os.getenv("REDIS_AUTH",""),  # if your Redis instance requires authentication
        ssl=False,  # Enterprise typically requires SSL
        ssl_cert_reqs="none",  # or "none" for self-signed certs
        decode_responses=False  # RedisSaver expects bytes
    )
   return RedisCache(redis_client,prefix=prefix)  
