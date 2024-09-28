from src.tools.rag_workflow import RAGWorkflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from asyncio import sleep

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str

w = RAGWorkflow()
w._timeout = 120.0

async def RAG_chat(w, query):
    retriever = await w.run(collection_name="cook_book")
    result = await w.run(query=query, retriever=retriever)
    async for chunk in result.async_response_gen():
        yield chunk

@app.post("/rag-chat")
async def root(user_query: UserQuery):
    return StreamingResponse(
        RAG_chat(w=w, query=user_query.query), 
        media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)