from fastapi import FastAPI
from app.api import admin, auth, query
import uvicorn
from app.config import settings

app = FastAPI(title="Adv RAG", description="Advanced RAG API", version="0.1.0-lesson-0")

app.include_router(admin.router)
app.include_router(auth.router)
app.include_router(query.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to Adv RAG API"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)