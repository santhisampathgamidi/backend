# main.py

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator
import asyncio
import uvicorn


# Auth
from auth import get_current_user, router as auth_router

# Import agents
from Leasee_agent import leasee_agent
from Lessor_agent import lessor_agent

app = FastAPI(title="AI Lease Agent Backend")

# Enable CORS so frontend (http://localhost:3000) can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth routes (e.g. /api/login)
app.include_router(auth_router, prefix="/api", tags=["auth"])

# Input schema
class ChatRequest(BaseModel):
    question: str


# Utility: stream text word-by-word
async def stream_response(text: str) -> AsyncGenerator[str, None]:
    for word in text.split():
        yield word + " "
        await asyncio.sleep(0.03)  # simulate streaming pace


@app.get("/")
def root():
    return {"message": "AI Lease Agent API is running 🚀"}


@app.post("/chat/leasee")
async def chat_leasee(req: ChatRequest, user=Depends(get_current_user)):
    if user["role"] != "leasee":
        return StreamingResponse(
            stream_response("❌ Unauthorized: Only leasees can access this endpoint"),
            media_type="text/plain",
        )

    result = leasee_agent.invoke({"question": req.question})
    return StreamingResponse(stream_response(result["answer"]), media_type="text/plain")


@app.post("/chat/lessor")
async def chat_lessor(req: ChatRequest, user=Depends(get_current_user)):
    if user["role"] != "lessor":
        return StreamingResponse(
            stream_response("❌ Unauthorized: Only lessors can access this endpoint"),
            media_type="text/plain",
        )

    result = lessor_agent.invoke({"question": req.question})
    return StreamingResponse(stream_response(result["answer"]), media_type="text/plain")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)