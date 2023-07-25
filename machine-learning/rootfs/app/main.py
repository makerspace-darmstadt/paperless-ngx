from contextlib import asynccontextmanager

from fastapi import FastAPI

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    app.state.classifier = 42
    yield
    # Clean up the ML models and release the resources
    app.state.classifier = None


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict")
async def predict(content: str):
    result = app.state.classifier
    return {"result": result}


@app.post("/train")
async def train(documents: list[str]):
    app.state.classifier = app.state.classifier + 1
    return {"result": app.state.classifier}
