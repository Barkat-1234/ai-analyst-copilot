import os
import re
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import NullPool
import pandas as pd
from decimal import Decimal
import json
from typing import List, Dict, Any, Optional
import time
import hashlib
from datetime import datetime, timedelta
from jose import JWTError, jwt

# Import monitoring
from monitoring import monitor

load_dotenv()

# ==================== AUTHENTICATION SETUP ====================

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))

security = HTTPBearer()

# Auth Models
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

# ==================== DATABASE USER AUTHENTICATION ====================

def get_user_from_db(email: str):
    """Get user from PostgreSQL database"""
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
    """Authenticate user from database"""
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
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
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
                detail=f"Role '{required_role}' required. You have '{current_user.role}'"
            )
        return current_user
    return role_checker

# ==================== MAIN APP ====================

# Setup PostgreSQL
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, poolclass=NullPool)

# Setup Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

app = FastAPI(title="AI Data Analyst Copilot", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sql_used: str
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]

# Cache
query_cache = {}

def get_cache_key(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()

def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```\s*', '', sql_text)
    sql_text = re.sub(r'`', '', sql_text)
    sql_text = re.sub(r'^sql\s*', '', sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r'\n+', ' ', sql_text)
    return sql_text.strip()

def convert_value(val):
    if val is None:
        return None
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, (list, dict)):
        return val
    if hasattr(val, 'isoformat'):
        return val.isoformat()
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore')
    return str(val)

def get_schema_info() -> Dict[str, Any]:
    inspector = inspect(engine)
    schema = {}
    for table_name in inspector.get_table_names():
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col['name'],
                "type": str(col['type']),
                "nullable": col['nullable'],
                "default": str(col['default']) if col['default'] else None
            })
        schema[table_name] = columns
    return schema

def format_schema_for_prompt(schema: Dict[str, Any]) -> str:
    result = ""
    for table, columns in schema.items():
        result += f"\nTable: {table}\n"
        result += "Columns:\n"
        for col in columns:
            result += f"  - {col['name']} ({col['type']})"
            if not col['nullable']:
                result += " NOT NULL"
            result += "\n"
    return result

# ==================== AUTH ENDPOINTS ====================

@app.post("/login", response_model=Token)
def login(request: LoginRequest):
    user = authenticate_user(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"]}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_email=user["email"],
        role=user["role"]
    )

@app.get("/protected")
def protected_route(current_user: TokenData = Depends(get_current_user)):
    return {
        "message": f"Hello {current_user.email}",
        "role": current_user.role,
        "access": "granted"
    }

@app.get("/admin-only")
def admin_route(current_user: TokenData = Depends(require_role("admin"))):
    return {"message": "Welcome Admin!"}

@app.get("/me")
def get_me(current_user: TokenData = Depends(get_current_user)):
    return {
        "email": current_user.email,
        "role": current_user.role
    }

# ==================== MONITORING ENDPOINT ====================

@app.get("/monitoring/stats")
def get_stats(current_user: TokenData = Depends(require_role("admin"))):
    """Get monitoring statistics (Admin only)"""
    return monitor.get_stats()

# ==================== PUBLIC ENDPOINTS ====================

@app.get("/")
def home():
    return {
        "status": "running", 
        "database": "PostgreSQL",
        "version": "2.0.0",
        "auth_required": True,
        "endpoints": ["/login", "/ask", "/health", "/schema", "/tables", "/protected", "/admin-only", "/me", "/monitoring/stats"]
    }

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/schema")
def get_schema(current_user: TokenData = Depends(get_current_user)):
    """Get database schema (requires authentication)"""
    try:
        schema = get_schema_info()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tables")
def get_tables(current_user: TokenData = Depends(get_current_user)):
    """List all tables (requires authentication)"""
    try:
        inspector = inspect(engine)
        return {"tables": inspector.get_table_names()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MAIN ASK ENDPOINT (PROTECTED) ====================

@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    current_user: TokenData = Depends(get_current_user)
):
    start_time = time.time()
    
    # Check cache
    cache_key = get_cache_key(req.question)
    if cache_key in query_cache:
        cached = query_cache[cache_key]
        cached["metadata"]["cached"] = True
        return cached
    
    try:
        # Get schema
        schema = get_schema_info()
        schema_text = format_schema_for_prompt(schema)
        
        # Generate SQL
        sql_prompt = f"""You are an expert SQL query generator for PostgreSQL.

Database Schema:
{schema_text}

User Question: {req.question}

Instructions:
1. Write ONLY the SQL query to answer this question
2. Use proper PostgreSQL syntax
3. DO NOT use backticks, ```sql, or markdown formatting
4. Output ONLY the SQL query text
5. Use table aliases for better readability
6. Include proper GROUP BY when using aggregations

SQL:"""
        
        sql_response = model.generate_content(sql_prompt)
        raw_sql = sql_response.text.strip()
        sql_query = clean_sql(raw_sql)
        
        print(f"\n{'='*50}")
        print(f"User: {current_user.email} ({current_user.role})")
        print(f"Question: {req.question}")
        print(f"SQL: {sql_query}")
        print(f"{'='*50}\n")
        
        # Execute SQL
        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()
            columns = list(result.keys())
            
            # Convert to dict list
            data = []
            for row in rows[:100]:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = convert_value(row[i])
                data.append(row_dict)
        
        # Generate explanation with IMPROVED FORMAT
        df = pd.DataFrame(data) if data else pd.DataFrame()
        
        # Create summary statistics
        summary = {}
        if not df.empty:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            for col in numeric_cols[:5]:
                summary[col] = {
                    "total": float(df[col].sum()),
                    "average": float(df[col].mean()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
        
        # IMPROVED PROMPT - Structured Business Response
        explain_prompt = f"""You are an expert Business Analyst. Analyze the data below and provide a structured business report.

QUESTION: {req.question}

SQL QUERY USED: {sql_query}

DATA SUMMARY:
- Total rows: {len(data)}
- Columns: {columns}

SAMPLE DATA (first 5 rows):
{data[:5] if data else 'No data'}

STATISTICS:
{json.dumps(summary, indent=2) if summary else 'No numeric columns found'}

Now provide your analysis in this EXACT format:

📊 ANSWER: (1 sentence answering the question directly with key numbers)

💡 INSIGHT: (1-2 sentences highlighting the most important finding. Which product/region/category performed best? Which performed worst? Include specific numbers)

📈 COMPARISON: (Compare the best vs worst. Calculate ratio if possible. Example: "X is Y times higher than Z")

🏢 BUSINESS INTERPRETATION: (What does this mean for the business? Which category is driving performance?)

🎯 RECOMMENDATION: (1 actionable business recommendation based on the data)

IMPORTANT: Use the actual numbers from the data above. Be specific. Do NOT make up numbers.

Your response:"""
        
        explanation = model.generate_content(explain_prompt)
        
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        response = {
            "answer": explanation.text,
            "sql_used": sql_query,
            "data": data,
            "metadata": {
                "row_count": len(data),
                "columns": columns,
                "query_time_ms": duration_ms,
                "cached": False,
                "user": current_user.email,
                "role": current_user.role
            }
        }
        
        # Log to monitoring
        monitor.log_request(
            user=current_user.email,
            question=req.question,
            sql=sql_query,
            duration_ms=duration_ms,
            status=200
        )
        
        # Cache results
        if len(query_cache) > 100:
            oldest_key = next(iter(query_cache))
            del query_cache[oldest_key]
        query_cache[cache_key] = response
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        duration_ms = round((time.time() - start_time) * 1000, 2)
        
        print(f"Error: {error_msg}")
        
        # Log error to monitoring
        monitor.log_request(
            user=current_user.email,
            question=req.question,
            sql="Error generating SQL",
            duration_ms=duration_ms,
            status=500
        )
        
        return {
            "answer": f"Sorry, I couldn't answer that question. Error: {error_msg[:200]}",
            "sql_used": "Error generating SQL",
            "data": [],
            "metadata": {
                "error": error_msg,
                "query_time_ms": duration_ms,
                "user": current_user.email
            }
        }

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )
