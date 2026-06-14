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
from decimal import Decimal
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import hashlib
from datetime import datetime, timedelta
from jose import JWTError, jwt
import uuid
import sqlglot
from sqlglot import parse_one, errors
import traceback

load_dotenv()

# ==================== CONFIGURATION ====================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))
MAX_ROWS_LIMIT = 500
MAX_ROWS_RETURN = 1000
FORBIDDEN_SQL_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']

# Request counter for monitoring
request_counter = 0
total_response_time = 0
error_counter = 0

# Database
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=10, max_overflow=20, pool_pre_ping=True)

# Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

app = FastAPI(title="AI Data Analyst Copilot", version="6.0.0")

# ==================== CORS MIDDLEWARE ====================

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

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

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
    """Get list of table names for better prompting"""
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

def format_currency(value: float) -> str:
    if value >= 1_000_000:
        return f"${value:,.2f}M"
    elif value >= 1_000:
        return f"${value:,.2f}"
    return f"${value:.2f}"

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
    """Health check endpoint - supports both GET and HEAD requests"""
    return {"status": "healthy", "version": "6.0.0"}

@app.options("/ask")
async def options_ask():
    """Handle CORS preflight requests"""
    return {"message": "OK"}

@app.post("/ask")
async def ask(
    req: AskRequest,
    current_user: TokenData = Depends(get_current_user)
):
    global request_counter, total_response_time, error_counter
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Increment request counter
    request_counter += 1
    
    try:
        # Get schema and tables
        schema = get_schema_info_cached()
        tables = list(schema.keys())
        schema_text = format_schema_for_prompt(schema)
        
        # Enhanced SQL prompt with table awareness
        sql_prompt = f"""Database Tables: {tables}

Schema:
{schema_text}

Question: {req.question}

Instructions:
1. Write ONLY a SELECT query
2. Use the exact table and column names from schema above
3. For monthly questions, use DATE_TRUNC('month', date_column) or EXTRACT
4. Return ONLY the SQL query, no explanations

SQL:"""
        
        sql_response = model.generate_content(sql_prompt)
        sql_query = clean_sql(sql_response.text.strip())
        
        print(f"Generated SQL: {sql_query}")
        
        # Validate
        is_valid, error_msg = validate_sql_readonly(sql_query)
        if not is_valid:
            error_counter += 1
            raise HTTPException(status_code=403, detail=error_msg)
        
        sql_query = enforce_limit(sql_query)
        
        # Execute
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
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        total_response_time += duration_ms
        
        # Generate metrics
        metrics = []
        if data:
            df = pd.DataFrame(data)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if 'revenue' in numeric_cols:
                total_revenue = df['revenue'].sum()
                metrics.append({
                    "key": "total_revenue",
                    "label": "Total Revenue",
                    "value": format_currency(total_revenue),
                    "icon": "💰",
                    "format_type": "currency"
                })
            
            metrics.append({
                "key": "record_count",
                "label": "Records",
                "value": str(len(data)),
                "icon": "📋",
                "format_type": "integer"
            })
        
        # Generate chart config
        chart = None
        if data and len(data) > 0:
            df = pd.DataFrame(data)
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if text_cols and numeric_cols:
                chart = {
                    "type": "bar",
                    "x_column": text_cols[0],
                    "y_column": numeric_cols[0],
                    "title": f"{numeric_cols[0]} by {text_cols[0]}"
                }
        
        # Generate answer
        if data:
            answer_prompt = f"""Question: {req.question}
Data summary: {len(data)} records found
First few rows: {data[:3]}

Provide a brief business answer (1-2 sentences):"""
            answer = model.generate_content(answer_prompt).text
        else:
            answer = f"No data found for: {req.question}"
        
        response = {
            "success": True,
            "answer": answer,
            "data": data,
            "metrics": metrics,
            "chart": chart,
            "metadata": {
                "row_count": len(data),
                "query_time_ms": duration_ms,
                "sql_used": sql_query,
                "user": current_user.email,
                "tables": tables
            },
            "request_id": request_id,
            "api_version": "6.0.0"
        }
        
        return response
        
    except HTTPException:
        error_counter += 1
        raise
    except Exception as e:
        error_counter += 1
        error_full = traceback.format_exc()
        print(f"ERROR in /ask: {error_full}")
        
        return {
            "success": False,
            "answer": f"Error: {str(e)[:200]}",
            "data": [],
            "metrics": [],
            "chart": None,
            "metadata": {
                "error": str(e),
                "traceback": error_full[:500],
                "sql_attempted": sql_query if 'sql_query' in locals() else "None"
            },
            "request_id": request_id,
            "api_version": "6.0.0"
        }

@app.get("/monitoring/stats")
async def get_monitoring_stats(current_user: TokenData = Depends(require_role("admin"))):
    global request_counter, total_response_time, error_counter
    
    avg_response_time = 0
    if request_counter > 0:
        avg_response_time = total_response_time / request_counter
    
    error_rate = 0
    if request_counter > 0:
        error_rate = (error_counter / request_counter) * 100
    
    return {
        "total_requests": request_counter,
        "avg_response_time_ms": round(avg_response_time, 2),
        "error_rate": round(error_rate, 2),
        "status": "healthy"
    }

@app.get("/tables")
async def get_tables(current_user: TokenData = Depends(get_current_user)):
    """List all tables in database"""
    tables = get_table_names()
    return {"tables": tables}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
