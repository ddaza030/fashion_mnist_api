# External libraries
import uvicorn
from fastapi import FastAPI, APIRouter, Response, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from modelo_fashion_mnist.predecir import predecir_fashion_mnist

app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediccion_fashion_mnist = APIRouter()


@prediccion_fashion_mnist.post('/prediccion_fashion_mnist', status_code=200)
async def prediccion_fashion_mnist(
        response: Response,
        file: UploadFile):

    try:
        imagen = file.file.read()
        data = predecir_fashion_mnist(imagen)

    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

    return data

app.include_router(prediccion_fashion_mnist)

if __name__ == '__main__':
    uvicorn.run(app)
