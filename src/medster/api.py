from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from medster.agent import Agent

# Load environment variables
load_dotenv()

app = FastAPI(title="Medster API", description="API for Medster Clinical Agent")

# Enable CORS
# Support both local development and production
allowed_origins = [
    "http://localhost:3000",  # Local development
    "https://medster-frontend-430915582144.us-central1.run.app",  # Cloud Run production
]

# Add additional production frontend URL from environment variable
production_frontend = os.getenv("FRONTEND_URL")
if production_frontend and production_frontend not in allowed_origins:
    allowed_origins.append(production_frontend)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent
# We might want to make this a dependency or singleton
agent = Agent()

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None # Not fully used yet, but good for future

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {"message": "Medster API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # The agent.run method currently takes a query string and returns a string
        # It handles its own internal task planning and execution
        response = agent.run(request.message)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
