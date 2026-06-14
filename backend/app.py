import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import QueuePool
import pandas as pd
import numpy as np
from decimal import Decimal
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import hashlib
from datetime import datetime, timedelta
from jose import JWTError, jwt
import uuid
import traceback

# Import RAG system - FIXED with dot
from rag import RAGSystem

load_dotenv()

# ==================== CONFIGURATION ====================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480
MAX_ROWS_LIMIT = 500
MAX_ROWS_RETURN = 100
FORBIDDEN_SQL_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']

# Request counter
request_counter = 0
total_response_time = 0
error_counter = 0

# Database
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=10, max_overflow=20, pool_pre_ping=True)

# Initialize RAG System
rag = RAGSystem()

# Gemini - Using gemini-2.0-flash
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.0-flash')

app = FastAPI(title="AI Data Analyst Copilot", version="7.0.0")

# ==================== CORS ====================

ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:8000",
    "https://data-analyst-copilot-ui.onrender.com",
    "https://ai-analyst-copilot-s5og.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ==================== 1. CHART/KPI DECISION ENGINE ====================

class ChartDecisionEngine:
    """Intelligently decides the best chart type based on data patterns"""
    
    @staticmethod
    def detect_data_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data patterns to recommend best visualization"""
        patterns = {
            "has_time_series": False,
            "has_categories": False,
            "has_numeric": False,
            "row_count": len(df),
            "numeric_columns": [],
            "categorical_columns": [],
            "time_columns": []
        }
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                patterns["numeric_columns"].append(col)
                patterns["has_numeric"] = True
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                patterns["time_columns"].append(col)
                patterns["has_time_series"] = True
            elif df[col].nunique() < 15:
                patterns["categorical_columns"].append(col)
                patterns["has_categories"] = True
        
        return patterns
    
    @staticmethod
    def decide_chart(df: pd.DataFrame, question: str, patterns: Dict) -> Dict[str, Any]:
        """Decide the best chart type based on data and question intent"""
        question_lower = question.lower()
        
        chart_config = {
            "type": "bar",
            "x_column": None,
            "y_column": None,
            "title": "Data Visualization",
            "recommended": True
        }
        
        if patterns["has_time_series"] and patterns["numeric_columns"]:
            chart_config["type"] = "line"
            chart_config["x_column"] = patterns["time_columns"][0]
            chart_config["y_column"] = patterns["numeric_columns"][0]
            chart_config["title"] = f"Trend of {patterns['numeric_columns'][0]} over time"
            
        elif patterns["has_categories"] and patterns["numeric_columns"]:
            chart_config["type"] = "bar"
            chart_config["x_column"] = patterns["categorical_columns"][0]
            chart_config["y_column"] = patterns["numeric_columns"][0]
            chart_config["title"] = f"{patterns['numeric_columns'][0]} by {patterns['categorical_columns'][0]}"
            
            if patterns["row_count"] <= 6 and "pie" in question_lower:
                chart_config["type"] = "pie"
                
        elif len(patterns["numeric_columns"]) >= 2:
            chart_config["type"] = "scatter"
            chart_config["x_column"] = patterns["numeric_columns"][0]
            chart_config["y_column"] = patterns["numeric_columns"][1]
            chart_config["title"] = f"{patterns['numeric_columns'][1]} vs {patterns['numeric_columns'][0]}"
        
        return chart_config

    @staticmethod
    def generate_kpi_metrics(df: pd.DataFrame, patterns: Dict) -> List[Dict]:
        """Generate relevant KPI metrics from data"""
        metrics = []
        
        metrics.append({
            "key": "total_records",
            "label": "Total Records",
            "value": str(len(df)),
            "icon": "📋"
        })
        
        for col in patterns["numeric_columns"][:3]:
            total = df[col].sum()
            avg = df[col].mean()
            
            is_currency = "revenue" in col.lower() or "price" in col.lower() or "sales" in col.lower()
            format_str = lambda x: f"${x:,.2f}" if is_currency else f"{x:,.0f}"
            
            metrics.append({
                "key": f"total_{col}",
                "label": f"Total {col.replace('_', ' ').title()}",
                "value": format_str(total),
                "icon": "💰" if is_currency else "📊"
            })
            
            metrics.append({
                "key": f"avg_{col}",
                "label": f"Avg {col.replace('_', ' ').title()}",
                "value": format_str(avg),
                "icon": "📈"
            })
        
        return metrics

# ==================== 2. INSIGHT GENERATION LAYER ====================

class InsightGenerator:
    """Generates business insights from data without exposing raw dataframes"""
    
    @staticmethod
    def generate_insights(df: pd.DataFrame, question: str, patterns: Dict) -> Dict[str, Any]:
        """Generate structured insights from data"""
        
        insights = {
            "summary": "",
            "key_findings": [],
            "recommendations": [],
            "anomalies": []
        }
        
        summary_parts = []
        summary_parts.append(f"Analysis of {len(df)} records")
        
        if patterns["numeric_columns"]:
            top_col = patterns["numeric_columns"][0]
            total = df[top_col].sum()
            avg = df[top_col].mean()
            summary_parts.append(f"Total {top_col}: {total:,.2f}")
            summary_parts.append(f"Average {top_col}: {avg:,.2f}")
        
        insights["summary"] = ", ".join(summary_parts)
        
        if patterns["categorical_columns"] and patterns["numeric_columns"]:
            cat_col = patterns["categorical_columns"][0]
            num_col = patterns["numeric_columns"][0]
            
            grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False)
            
            if len(grouped) > 0:
                top_item = grouped.index[0]
                top_value = grouped.iloc[0]
                insights["key_findings"].append(f"Top performer: {top_item} with {top_value:,.2f}")
                
                if len(grouped) > 1:
                    bottom_item = grouped.index[-1]
                    bottom_value = grouped.iloc[-1]
                    insights["key_findings"].append(f"Lowest performer: {bottom_item} with {bottom_value:,.2f}")
                    
                    ratio = top_value / bottom_value if bottom_value > 0 else 0
                    if ratio > 2:
                        insights["key_findings"].append(f"Top performer is {ratio:.1f}x higher than lowest performer")
        
        if patterns["numeric_columns"]:
            num_col = patterns["numeric_columns"][0]
            if df[num_col].sum() > 10000:
                insights["recommendations"].append(f"Strong total {num_col} - consider increasing investment")
            if df[num_col].mean() > df[num_col].median():
                insights["recommendations"].append("High-value outliers detected - review top performers")
        
        if patterns["numeric_columns"]:
            num_col = patterns["numeric_columns"][0]
            mean_val = df[num_col].mean()
            std_val = df[num_col].std()
            anomalies = df[df[num_col] > mean_val + 2 * std_val]
            
            if len(anomalies) > 0:
                insights["anomalies"].append(f"Found {len(anomalies)} unusually high values")
        
        return insights

# ==================== SQL FUNCTIONS ====================

def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```\s*', '', sql_text)
    sql_text = re.sub(r'`', '', sql_text)
    sql_text = re.sub(r'^sql\s*', '', sql_text, flags=re.IGNORECASE)
    sql_text = sql_text.replace(';', '')
    return sql_text.strip()

def enforce_limit(sql_query: str) -> str:
    sql_query = sql_query.rstrip(';').strip()
    if 'LIMIT' not in sql_query.upper():
        sql_query = f"{sql_query} LIMIT {MAX_ROWS_LIMIT}"
    return sql_query

def validate_sql_readonly(sql_query: str) -> Tuple[bool, str]:
    sql_upper = sql_query.upper()
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"Forbidden keyword: {keyword}"
    if not sql_upper.strip().startswith('SELECT'):
        return False, "Only SELECT queries allowed"
    return True, "OK"

def get_table_names() -> List[str]:
    inspector = inspect(engine)
    return inspector.get_table_names()

# ==================== AUTHENTICATION ====================

security = HTTPBearer()

class LoginRequest(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_email: str
    role: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None

def get_user_from_db(email: str):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT email, role, password FROM users WHERE email = :email"),
                {"email": email}
            )
            row = result.fetchone()
            if row:
                return {"email": row[0], "role": row[1], "password": row[2]}
            return None
    except Exception as e:
        print(f"Database error: {e}")
        return None

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, stored_password: str) -> bool:
    if plain_password == stored_password:
        return True
    return hash_password(plain_password) == stored_password

def authenticate_user(email: str, password: str):
    user = get_user_from_db(email)
    if user and verify_password(password, user["password"]):
        return user
    return None

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        role = payload.get("role")
        if email is None:
            raise credentials_exception
        return TokenData(email=email, role=role)
    except JWTError:
        raise credentials_exception

def require_role(required_role: str):
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required"
            )
        return current_user
    return role_checker

# ==================== SCHEMA CACHING ====================

_schema_cache = None
_schema_cache_time = 0

def get_schema_info_cached():
    global _schema_cache, _schema_cache_time
    current_time = time.time()
    if _schema_cache is None or (current_time - _schema_cache_time) > 3600:
        inspector = inspect(engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    "name": col['name'],
                    "type": str(col['type']),
                    "nullable": col['nullable']
                })
            schema[table_name] = columns
        _schema_cache = schema
        _schema_cache_time = current_time
    return _schema_cache

def format_schema_for_prompt(schema):
    result = ""
    for table, columns in schema.items():
        result += f"\nTable: {table}\nColumns:\n"
        for col in columns:
            result += f"  - {col['name']} ({col['type']})\n"
    return result

# ==================== HELPER FUNCTIONS ====================

def convert_value(val):
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (int, float)):
        return val
    if hasattr(val, 'isoformat'):
        return val.isoformat()
    return str(val)

# ==================== API MODELS ====================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

# ==================== ENDPOINTS ====================

@app.post("/login", response_model=Token)
def login(login_req: LoginRequest):
    user = authenticate_user(login_req.email, login_req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_email=user["email"],
        role=user["role"]
    )

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "healthy", "version": "7.0.0"}

@app.options("/ask")
async def options_ask():
    return {"message": "OK"}

@app.post("/ask")
async def ask(
    req: AskRequest,
    current_user: TokenData = Depends(get_current_user)
):
    global request_counter, total_response_time, error_counter
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    request_counter += 1
    
    try:
        # ==================== RAG STEP 1: SEARCH KNOWLEDGE BASE ====================
        rag_results = rag.search(req.question, top_k=3)
        
        # Build context from RAG results
        rag_context = ""
        if rag_results:
            rag_context = "\n\nRelevant Knowledge from Database:\n"
            for i, result in enumerate(rag_results, 1):
                rag_context += f"{i}. {result['content']}\n"
        
        schema = get_schema_info_cached()
        tables = list(schema.keys())
        schema_text = format_schema_for_prompt(schema)
        
        # ==================== RAG STEP 2: AUGMENT PROMPT WITH CONTEXT ====================
        sql_prompt = f"""Tables: {tables}
Schema:
{schema_text}
{rag_context}

Question: {req.question}

Use the schema above and any relevant knowledge to write the SQL query.
SQL:"""
        
        sql_response = model.generate_content(sql_prompt)
        sql_query = clean_sql(sql_response.text.strip())
        
        print(f"Generated SQL: {sql_query}")
        print(f"RAG Context Used: {len(rag_results)} documents retrieved")
        
        is_valid, error_msg = validate_sql_readonly(sql_query)
        if not is_valid:
            error_counter += 1
            raise HTTPException(status_code=403, detail=error_msg)
        
        sql_query = enforce_limit(sql_query)
        
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = list(result.keys())
            
            data = []
            for row in rows[:MAX_ROWS_RETURN]:
                row_dict = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    elif isinstance(val, Decimal):
                        val = float(val)
                    row_dict[col] = val
                data.append(row_dict)
        
        if not data:
            return {
                "success": True,
                "answer": "No data found",
                "metrics": [],
                "chart": None,
                "insights": {"summary": "No data available"},
                "rag_used": len(rag_results) > 0,
                "data": [],
                "metadata": {"row_count": 0}
            }
        
        df = pd.DataFrame(data)
        
        # Detect patterns
        patterns = ChartDecisionEngine.detect_data_patterns(df)
        
        # Generate chart config
        chart_config = ChartDecisionEngine.decide_chart(df, req.question, patterns)
        
        # Generate KPI metrics
        kpi_metrics = ChartDecisionEngine.generate_kpi_metrics(df, patterns)
        
        # Generate insights
        insights = InsightGenerator.generate_insights(df, req.question, patterns)
        
        # ==================== RAG STEP 3: USE RAG CONTEXT IN ANSWER ====================
        answer_prompt = f"""Question: {req.question}
Insights: {json.dumps(insights, indent=2)}
{rag_context}

Provide a concise business answer (2-3 sentences). If the knowledge base has relevant information, incorporate it:"""
        
        answer = model.generate_content(answer_prompt).text
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        total_response_time += duration_ms
        
        response = {
            "success": True,
            "answer": answer,
            "metrics": kpi_metrics,
            "chart": chart_config,
            "insights": insights,
            "rag_used": len(rag_results) > 0,
            "rag_sources": [r["content"][:200] for r in rag_results] if rag_results else [],
            "data": data[:20],
            "metadata": {
                "row_count": len(data),
                "query_time_ms": duration_ms,
                "sql_used": sql_query,
                "user": current_user.email
            },
            "request_id": request_id,
            "api_version": "7.0.0"
        }
        
        return response
        
    except HTTPException:
        error_counter += 1
        raise
    except Exception as e:
        error_counter += 1
        error_full = traceback.format_exc()
        print(f"ERROR: {error_full}")
        
        return {
            "success": False,
            "answer": f"Error: {str(e)[:200]}",
            "metrics": [],
            "chart": None,
            "insights": {},
            "rag_used": False,
            "data": [],
            "metadata": {"error": str(e)},
            "request_id": request_id,
            "api_version": "7.0.0"
        }

@app.get("/monitoring/stats")
async def get_monitoring_stats(current_user: TokenData = Depends(require_role("admin"))):
    global request_counter, total_response_time, error_counter
    
    avg_response_time = round(total_response_time / request_counter, 2) if request_counter > 0 else 0
    error_rate = round((error_counter / request_counter) * 100, 2) if request_counter > 0 else 0
    
    return {
        "total_requests": request_counter,
        "avg_response_time_ms": avg_response_time,
        "error_rate": error_rate,
        "status": "healthy"
    }

@app.get("/tables")
async def get_tables(current_user: TokenData = Depends(get_current_user)):
    tables = get_table_names()
    return {"tables": tables}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
