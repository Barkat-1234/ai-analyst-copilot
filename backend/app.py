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
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from decimal import Decimal
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import hashlib
from datetime import datetime, timedelta
from jose import JWTError, jwt
import uuid
import asyncio
import sqlglot
from sqlglot import parse_one, errors
from functools import lru_cache
from contextlib import contextmanager

# Note: redis and slowapi are optional - commenting out for simpler deployment
# import redis
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded

load_dotenv()

# ==================== CONFIGURATION ====================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

# Security Configuration
MAX_QUERY_TIMEOUT_SECONDS = 30
MAX_ROWS_RETURN = 1000
MAX_ROWS_LIMIT = 500
FORBIDDEN_SQL_KEYWORDS = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE', 'MERGE', 'REPLACE']

# Cache Configuration
SCHEMA_CACHE_TTL = 3600  # 1 hour
QUERY_CACHE_MAX_SIZE = 100
QUERY_CACHE_TTL = 300  # 5 minutes

# PostgreSQL Setup with Connection Pooling
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(
    DATABASE_URL, 
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Gemini Setup
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# ==================== FASTAPI APP ====================

app = FastAPI(title="AI Data Analyst Copilot", version="4.0.0")

# ==================== CORS MIDDLEWARE (FIXED - CRITICAL) ====================

# Define allowed origins - your frontend URLs
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:8000",
    "https://ai-analyst-frontend-2k26.onrender.com",
    "https://ai-analyst.onrender.com",
    "https://ai-analyst-copilot-2.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],  # OPTIONS is CRITICAL for CORS
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allow all hosts for now
)

# ==================== SQL VALIDATION ====================

def validate_sql_ast(sql_query: str) -> Tuple[bool, str]:
    """Use sqlglot to validate SQL syntax and structure"""
    try:
        parsed = parse_one(sql_query, dialect="postgres")
        
        # Check if it's a SELECT statement
        first_keyword = str(parsed).upper().split()[0] if parsed else ""
        if first_keyword != 'SELECT':
            return False, "Only SELECT statements are allowed"
        
        return True, "Valid SQL"
    except errors.ParseError as e:
        return False, f"SQL syntax error: {str(e)}"

def enforce_limit(sql_query: str) -> str:
    """Enforce row limit on SQL queries"""
    sql_upper = sql_query.upper()
    
    if 'LIMIT' in sql_upper:
        import re
        pattern = r'LIMIT\s+(\d+)'
        match = re.search(pattern, sql_upper, re.IGNORECASE)
        if match:
            current_limit = int(match.group(1))
            if current_limit > MAX_ROWS_LIMIT:
                sql_query = re.sub(pattern, f'LIMIT {MAX_ROWS_LIMIT}', sql_query, flags=re.IGNORECASE)
    else:
        sql_query = f"{sql_query} LIMIT {MAX_ROWS_LIMIT}"
    
    return sql_query

def validate_sql_readonly(sql_query: str) -> Tuple[bool, str]:
    """Validate SQL is read-only"""
    sql_upper = sql_query.upper()
    
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"SQL contains forbidden keyword: {keyword}"
    
    return True, "OK"

# ==================== CACHE MANAGEMENT ====================

class PerUserCache:
    def __init__(self, max_size: int = QUERY_CACHE_MAX_SIZE, ttl_seconds: int = QUERY_CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict] = {}
        self._schema_hash = None
    
    def update_schema_hash(self, schema_hash: str):
        self._schema_hash = schema_hash
        self._cache.clear()
    
    def _get_user_key(self, user_email: str, question: str, schema_hash: str) -> str:
        return hashlib.md5(f"{user_email}:{question.lower().strip()}:{schema_hash}".encode()).hexdigest()
    
    def get(self, user_email: str, question: str, schema_hash: str) -> Optional[Dict]:
        key = self._get_user_key(user_email, question, schema_hash)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            else:
                del self._cache[key]
        return None
    
    def set(self, user_email: str, question: str, schema_hash: str, data: Dict) -> None:
        key = self._get_user_key(user_email, question, schema_hash)
        
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
        
        self._cache[key] = {
            'data': data,
            'timestamp': time.time(),
            'user': user_email
        }

query_cache = PerUserCache()

# ==================== SCHEMA CACHING ====================

_schema_cache = None
_schema_cache_time = 0
_schema_hash = None

def get_schema_hash(schema: Dict[str, Any]) -> str:
    schema_str = json.dumps(schema, sort_keys=True)
    return hashlib.md5(schema_str.encode()).hexdigest()

def get_schema_info_cached() -> Tuple[Dict[str, Any], str]:
    global _schema_cache, _schema_cache_time, _schema_hash
    
    current_time = time.time()
    if _schema_cache is None or (current_time - _schema_cache_time) > SCHEMA_CACHE_TTL:
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
        _schema_hash = get_schema_hash(schema)
        query_cache.update_schema_hash(_schema_hash)
    
    return _schema_cache, _schema_hash

def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    result = ""
    for table, columns in schema.items():
        result += f"\nTable: {table}\nColumns:\n"
        for col in columns:
            result += f"  - {col['name']} ({col['type']})"
            if not col['nullable']:
                result += " NOT NULL"
            result += "\n"
    return result

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

def authenticate_user(email: str, password: str):
    user = get_user_from_db(email)
    if user and user["password"] == password:
        return user
    return None

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
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

# ==================== HELPER FUNCTIONS ====================

def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```\s*', '', sql_text)
    sql_text = re.sub(r'`', '', sql_text)
    sql_text = re.sub(r'^sql\s*', '', sql_text, flags=re.IGNORECASE)
    return sql_text.strip()

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
def login(request: Request, login_req: LoginRequest):
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

@app.get("/health")
def health():
    return {"status": "healthy", "version": "4.0.0"}

@app.post("/ask")
async def ask(
    request: Request,
    req: AskRequest,
    current_user: TokenData = Depends(get_current_user)
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Get cached schema with hash
        schema, schema_hash = get_schema_info_cached()
        schema_text = format_schema_for_prompt(schema)
        
        # Check cache
        cached = query_cache.get(current_user.email, req.question, schema_hash)
        if cached:
            cached["metadata"]["cached"] = True
            cached["request_id"] = request_id
            return cached
        
        # Generate SQL
        sql_prompt = f"""Database Schema:
{schema_text}

Question: {req.question}

Write ONLY the SQL query (SELECT only):"""
        
        sql_response = model.generate_content(sql_prompt)
        sql_query = clean_sql(sql_response.text.strip())
        
        # Security validations
        is_valid_ro, ro_error = validate_sql_readonly(sql_query)
        if not is_valid_ro:
            raise HTTPException(status_code=403, detail=ro_error)
        
        # Enforce row limit
        sql_query = enforce_limit(sql_query)
        
        # Execute query
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = list(result.keys())
            
            # Convert to dict list
            data = []
            for row in rows[:MAX_ROWS_RETURN]:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = convert_value(row[i])
                data.append(row_dict)
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
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
        
        response = {
            "success": True,
            "answer": f"Found {len(data)} records",
            "sql_used": sql_query,
            "sql_explanation": "Query executed successfully",
            "data": data,
            "metrics": metrics,
            "chart": chart,
            "formatting_rules": [],
            "query_cost": {"complexity": "low", "score": 1},
            "metadata": {
                "row_count": len(data),
                "query_time_ms": duration_ms,
                "cached": False,
                "user": current_user.email
            },
            "request_id": request_id,
            "api_version": "4.0.0"
        }
        
        query_cache.set(current_user.email, req.question, schema_hash, response)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        return {
            "success": False,
            "answer": f"Error: {str(e)[:200]}",
            "sql_used": "",
            "sql_explanation": "",
            "data": [],
            "metrics": [],
            "chart": None,
            "formatting_rules": [],
            "query_cost": {},
            "metadata": {"error": str(e)},
            "request_id": request_id,
            "api_version": "4.0.0"
        }

@app.get("/monitoring/stats")
async def get_monitoring_stats(current_user: TokenData = Depends(require_role("admin"))):
    return {
        "total_requests": 0,
        "avg_response_time_ms": 0,
        "error_rate": 0,
        "status": "healthy"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
