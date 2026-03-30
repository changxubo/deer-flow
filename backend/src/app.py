import uvicorn
from src.api.server import app

if __name__ == "__main__":
    import uvicorn
    # Using the same port as original app.py
    uvicorn.run("src.app:app", host="0.0.0.0", port=2024, reload=True)
