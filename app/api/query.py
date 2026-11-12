from fastapi import APIRouter, Depends 
from app.middleware.auth import User, get_current_user
from app.models import ChatResponse, QueryRequest
from app.services.rag_service import run_rag_with_trace



router = APIRouter(tags=["query"])

@router.post("/query", response_model=ChatResponse)
async def query(body: QueryRequest, user: User = Depends(get_current_user)) -> ChatResponse:
    response, _chunks = run_rag_with_trace(body.question, flags={"top_k": body.top_k})
    return response
