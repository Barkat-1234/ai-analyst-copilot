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
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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

# Rate Limiting
RATE_LIMIT = "20/minute"
RATE_LIMIT_ADMIN = "100/minute"

# Cache Configuration
SCHEMA_CACHE_TTL = 3600  # 1 hour
QUERY_CACHE_MAX_SIZE = 100
QUERY_CACHE_TTL = 300  # 5 minutes

# Redis (optional - fallback to memory if not available)
REDIS_URL = os.getenv("REDIS_URL", None)
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_available = True
    else:
        redis_available = False
except:
    redis_available = False

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

# ==================== RATE LIMITING ====================

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI Data Analyst Copilot", version="4.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ==================== MIDDLEWARE ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# ==================== SQL AST VALIDATION (sqlglot) ====================

def validate_sql_ast(sql_query: str) -> Tuple[bool, str]:
    """Use sqlglot to validate SQL syntax and structure"""
    try:
        parsed = parse_one(sql_query, dialect="postgres")
        
        # Check if it's a SELECT statement
        if parsed.keywords and parsed.keywords[0].upper() != 'SELECT':
            return False, "Only SELECT statements are allowed"
        
        # Extract limit to enforce caps
        limit = None
        for node in parsed.walk():
            if hasattr(node, 'args') and 'limit' in node.args:
                limit = node.args['limit']
                break
        
        return True, "Valid SQL"
    except errors.ParseError as e:
        return False, f"SQL syntax error: {str(e)}"

def enforce_limit(sql_query: str) -> str:
    """Enforce row limit on SQL queries"""
    sql_upper = sql_query.upper()
    
    # Check if LIMIT already exists
    if 'LIMIT' in sql_upper:
        # Replace existing limit if it's too high
        import re
        pattern = r'LIMIT\s+(\d+)'
        match = re.search(pattern, sql_upper, re.IGNORECASE)
        if match:
            current_limit = int(match.group(1))
            if current_limit > MAX_ROWS_LIMIT:
                sql_query = re.sub(pattern, f'LIMIT {MAX_ROWS_LIMIT}', sql_query, flags=re.IGNORECASE)
    else:
        # Add limit
        sql_query = f"{sql_query} LIMIT {MAX_ROWS_LIMIT}"
    
    return sql_query

def validate_sql_readonly(sql_query: str) -> Tuple[bool, str]:
    """Validate SQL is read-only (no destructive operations)"""
    sql_upper = sql_query.upper()
    
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return False, f"SQL contains forbidden keyword: {keyword}"
    
    return True, "OK"

def validate_query_timeout(start_time: float) -> bool:
    """Check if query has exceeded timeout"""
    elapsed = time.time() - start_time
    if elapsed > MAX_QUERY_TIMEOUT_SECONDS:
        return False
    return True

# ==================== SANITIZED ERROR RESPONSES ====================

def sanitize_error_response(error: Exception, user_role: str) -> Dict[str, Any]:
    """Return sanitized error messages (no internal details to non-admins)"""
    error_type = type(error).__name__
    
    if user_role == "admin":
        return {
            "error_type": error_type,
            "message": str(error)[:200]
        }
    else:
        # Generic message for non-admins
        if "syntax" in str(error).lower():
            return {
                "error_type": "QueryError",
                "message": "There was an issue processing your query. Please rephrase your question."
            }
        elif "permission" in str(error).lower() or "forbidden" in str(error).lower():
            return {
                "error_type": "PermissionError",
                "message": "You don't have permission to perform this action."
            }
        else:
            return {
                "error_type": "InternalError",
                "message": "An unexpected error occurred. Please try again later."
            }

# ==================== PROMPT HARDENING ====================

SYSTEM_PROMPT = """
You are an AI Data Analyst for a PostgreSQL database. Follow these rules STRICTLY:

RULES:
1. Generate ONLY SELECT queries
2. DO NOT use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE
3. DO NOT make up tables or columns
4. Use ONLY the schema provided
5. If the question cannot be answered with available schema, respond with "UNABLE_TO_ANSWER"
6. Keep queries simple and efficient
7. Always add LIMIT clause
"""

def get_sql_prompt(schema_text: str, question: str) -> str:
    """Get hardened SQL prompt"""
    return f"""{SYSTEM_PROMPT}

DATABASE SCHEMA:
{schema_text}

USER QUESTION: {question}

Generate ONLY the SQL query:"""

# ==================== CACHE MANAGEMENT ====================

class PerUserCache:
    """Cache isolated per user with schema version tracking"""
    
    def __init__(self, max_size: int = QUERY_CACHE_MAX_SIZE, ttl_seconds: int = QUERY_CACHE_TTL):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[str, Dict] = {}
        self._schema_hash = None
    
    def update_schema_hash(self, schema_hash: str):
        """Update schema hash to invalidate cache when schema changes"""
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

# Initialize cache
query_cache = PerUserCache()

# ==================== SCHEMA CACHING WITH HASH ====================

_schema_cache = None
_schema_cache_time = 0
_schema_hash = None

def get_schema_hash(schema: Dict[str, Any]) -> str:
    """Generate hash of schema for cache invalidation"""
    schema_str = json.dumps(schema, sort_keys=True)
    return hashlib.md5(schema_str.encode()).hexdigest()

def get_schema_info_cached() -> Tuple[Dict[str, Any], str]:
    """Get cached database schema with hash"""
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

# ==================== QUERY COST ESTIMATOR ====================

def estimate_query_cost(sql_query: str) -> Dict[str, Any]:
    """Estimate query complexity cost"""
    sql_lower = sql_query.lower()
    
    cost = {
        "complexity": "low",
        "score": 1,
        "estimated_time_ms": 100,
        "warnings": []
    }
    
    # Check for expensive operations
    if "join" in sql_lower:
        cost["score"] += 2
        cost["warnings"].append("Query uses JOIN operations")
    
    if "group by" in sql_lower:
        cost["score"] += 2
        cost["warnings"].append("Query uses GROUP BY")
    
    if "order by" in sql_lower:
        cost["score"] += 1
    
    if "distinct" in sql_lower:
        cost["score"] += 1
    
    if "where" in sql_lower:
        cost["score"] += 1
    
    # Determine complexity level
    if cost["score"] <= 3:
        cost["complexity"] = "low"
        cost["estimated_time_ms"] = 100
    elif cost["score"] <= 6:
        cost["complexity"] = "medium"
        cost["estimated_time_ms"] = 500
    else:
        cost["complexity"] = "high"
        cost["estimated_time_ms"] = 2000
    
    return cost

# ==================== ASYNC DB + GEMINI ====================

async def async_generate_content(prompt: str) -> str:
    """Async wrapper for Gemini API"""
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt)
    return response.text

async def async_execute_sql(sql_query: str) -> Tuple[List, List]:
    """Async wrapper for SQL execution"""
    loop = asyncio.get_event_loop()
    
    def execute():
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            return result.fetchall(), list(result.keys())
    
    return await loop.run_in_executor(None, execute)

# ==================== OBSERVABILITY METRICS ====================

class MetricsCollector:
    """Separate metrics for LLM vs DB performance"""
    
    def __init__(self):
        self.llm_total_time = 0
        self.db_total_time = 0
        self.llm_calls = 0
        self.db_calls = 0
    
    def record_llm(self, duration_ms: float):
        self.llm_total_time += duration_ms
        self.llm_calls += 1
    
    def record_db(self, duration_ms: float):
        self.db_total_time += duration_ms
        self.db_calls += 1
    
    def get_stats(self):
        return {
            "llm_avg_ms": round(self.llm_total_time / self.llm_calls, 2) if self.llm_calls else 0,
            "db_avg_ms": round(self.db_total_time / self.db_calls, 2) if self.db_calls else 0,
            "llm_calls": self.llm_calls,
            "db_calls": self.db_calls
        }

metrics_collector = MetricsCollector()

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

def format_number(value: float) -> str:
    return f"{int(value):,}" if value == int(value) else f"{value:,.2f}"

# ==================== API MODELS ====================

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class AskResponse(BaseModel):
    success: bool
    answer: str
    sql_used: str
    sql_explanation: str
    data: List[Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    chart: Optional[Dict[str, Any]]
    formatting_rules: List[Dict[str, Any]]
    query_cost: Dict[str, Any]
    metadata: Dict[str, Any]
    request_id: str
    api_version: str

# ==================== ENDPOINTS ====================

@app.post("/login", response_model=Token)
@limiter.limit(RATE_LIMIT)
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
@limiter.limit(RATE_LIMIT)
async def ask(
    request: Request,
    req: AskRequest,
    current_user: TokenData = Depends(get_current_user)
):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Get rate limit for user role
    if current_user.role == "admin":
        limiter.limit(RATE_LIMIT_ADMIN)(ask)
    
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
        
        # Generate SQL with hardened prompt
        llm_start = time.time()
        sql_prompt = get_sql_prompt(schema_text, req.question)
        sql_response = await async_generate_content(sql_prompt)
        llm_duration = (time.time() - llm_start) * 1000
        metrics_collector.record_llm(llm_duration)
        
        sql_query = clean_sql(sql_response.strip())
        
        # AST Validation
        is_valid_ast, ast_error = validate_sql_ast(sql_query)
        if not is_valid_ast:
            raise HTTPException(status_code=400, detail=ast_error)
        
        # Read-only validation
        is_valid_ro, ro_error = validate_sql_readonly(sql_query)
        if not is_valid_ro:
            raise HTTPException(status_code=403, detail=ro_error)
        
        # Enforce row limit
        sql_query = enforce_limit(sql_query)
        
        # Estimate query cost
        query_cost = estimate_query_cost(sql_query)
        
        # Execute query
        db_start = time.time()
        rows, columns = await async_execute_sql(sql_query)
        db_duration = (time.time() - db_start) * 1000
        metrics_collector.record_db(db_duration)
        
        # Process results
        data = []
        for row in rows[:MAX_ROWS_RETURN]:
            row_dict = {}
            for i, col in enumerate(columns):
                row_dict[col] = convert_value(row[i])
            data.append(row_dict)
        
        # Generate response
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        response = {
            "success": True,
            "answer": f"Found {len(data)} records",
            "sql_used": sql_query,
            "sql_explanation": "Query executed successfully",
            "data": data,
            "metrics": [],
            "chart": None,
            "formatting_rules": [],
            "query_cost": query_cost,
            "metadata": {
                "row_count": len(data),
                "query_time_ms": duration_ms,
                "llm_time_ms": llm_duration,
                "db_time_ms": db_duration,
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
        error_response = sanitize_error_response(e, current_user.role)
        return {
            "success": False,
            "answer": error_response["message"],
            "sql_used": "",
            "sql_explanation": "",
            "data": [],
            "metrics": [],
            "chart": None,
            "formatting_rules": [],
            "query_cost": {},
            "metadata": {"error": error_response.get("error_type", "Unknown")},
            "request_id": request_id,
            "api_version": "4.0.0"
        }

@app.get("/metrics/stats")
@limiter.limit(RATE_LIMIT_ADMIN)
async def get_metrics(current_user: TokenData = Depends(require_role("admin"))):
    """Get observability metrics (admin only)"""
    return {
        "llm_performance": {
            "avg_ms": metrics_collector.llm_total_time / metrics_collector.llm_calls if metrics_collector.llm_calls else 0,
            "total_calls": metrics_collector.llm_calls
        },
        "db_performance": {
            "avg_ms": metrics_collector.db_total_time / metrics_collector.db_calls if metrics_collector.db_calls else 0,
            "total_calls": metrics_collector.db_calls
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
