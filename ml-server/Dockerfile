FROM python:3.11-slim

WORKDIR /ml_server

COPY ./app /ml_server/app
COPY requirements.txt /ml_server

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]