FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Don't copy .env - use Render environment variables instead

EXPOSE 8000 8501

CMD sh -c "cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT & cd frontend && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"