# AI Analyst Copilot

An AI-powered business intelligence assistant that converts **natural language into SQL queries**, retrieves data from PostgreSQL, and generates AI-powered insights with interactive visualizations.

## Features

* 💬 Natural Language to SQL
* 🗄️ PostgreSQL database integration
* 🤖 Groq LLM for SQL generation & analytics
* 📚 RAG-powered business knowledge retrieval
* 📊 Interactive charts and dashboards
* 🔐 JWT authentication
* ⚡ FastAPI backend
* 🎨 Streamlit frontend
* 🐳 Docker deployment

## Tech Stack

* Python
* FastAPI
* Streamlit
* Groq API
* PostgreSQL
* ChromaDB
* LangChain
* SQLAlchemy
* JWT Authentication
* Docker

## Installation

```bash
git clone https://github.com/Barkat-1234/ai-analyst-copilot.git
cd ai-analyst-copilot

python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Run the application:

```bash
docker compose up --build
```

or run the backend and frontend separately.

## Project Structure

```text
backend/        # FastAPI APIs & AI services
frontend/       # Streamlit user interface
Dockerfile
requirements.txt
```

## Workflow

1. User asks a business question in plain English.
2. AI converts the question into SQL.
3. PostgreSQL executes the query.
4. Retrieved data is analyzed by the LLM.
5. Interactive charts and business insights are generated.

## License

This project is intended for learning, research, and portfolio demonstration purposes.
