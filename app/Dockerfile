FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY notebooks/08_advanced_modeling.py .
COPY src/models/ ./src/models/
COPY app/app.py .

RUN mkdir -p reports/figures data/processed

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
