from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import router
from app.core.config import settings
from app.services.rag_service import rag_service
from app.core.logging import logger

app = FastAPI(title=settings.app_name, version=settings.app_version)

app.include_router(router)
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    await rag_service.initialize()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(app, host=settings.host, port=settings.port)
