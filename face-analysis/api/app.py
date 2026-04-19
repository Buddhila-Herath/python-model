from fastapi import FastAPI

from api.routes import router


app = FastAPI(title="Face Analysis API", version="1.0.0")
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
