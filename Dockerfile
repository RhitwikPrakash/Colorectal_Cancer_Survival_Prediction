FROM python:3.9-slim

WORKDIR /app

COPY requirements-mlops.txt .
RUN pip install -r requirements-mlops.txt

COPY src/ /app/src/                  

ENTRYPOINT ["python"]                 
