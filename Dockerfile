FROM python:3.11

WORKDIR opt/fashion_mnist_api

COPY ./fashion_mnist_api .
RUN pip install -r requirements.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"] 