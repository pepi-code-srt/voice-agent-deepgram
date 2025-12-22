from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Startup: Pre-warm Deepgram tokens
    print("Initializing Deepgram tokens...")
    yield
    # Shutdown
    print("Closing connections...")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"status": "Voice Agent Running"}
