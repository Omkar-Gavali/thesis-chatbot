from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_service
from app.core.logging import logger

router = APIRouter()

class Query(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@router.post("/query", response_model=Answer)
async def query(query: Query):
    try:
        logger.info(f"Received query: {query.question}")
        answer = await rag_service.query(query.question)
        return Answer(answer=answer)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
