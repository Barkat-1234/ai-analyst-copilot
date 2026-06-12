# backend/auth.py
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours

# Security scheme
security = HTTPBearer()

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    user_email: str
    role: str

class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class User(BaseModel):
    email: str
    role: str
    password: str  # Plain text for development

# Mock user database (Replace with your actual users table in production)
# Passwords are plain text for development - USE HASHING IN PRODUCTION
USERS_DB = {
    "admin@company.com": {
        "email": "admin@company.com",
        "role": "admin",
        "password": "admin123"
    },
    "analyst@company.com": {
        "email": "analyst@company.com",
        "role": "analyst",
        "password": "analyst123"
    },
    "viewer@company.com": {
        "email": "viewer@company.com",
        "role": "viewer",
        "password": "viewer123"
    }
}

def verify_password(plain_password: str, stored_password: str) -> bool:
    """Verify password (simple version for development)"""
    return plain_password == stored_password

def get_user(email: str) -> Optional[User]:
    """Get user from database"""
    if email in USERS_DB:
        user_data = USERS_DB[email]
        return User(
            email=user_data["email"],
            role=user_data["role"],
            password=user_data["password"]
        )
    return None

def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user"""
    user = get_user(email)
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        role: str = payload.get("role")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email, role=role)
    except JWTError:
        raise credentials_exception
    
    return token_data

def require_role(required_role: str):
    """Role-based access control decorator"""
    async def role_checker(current_user: TokenData = Depends(get_current_user)):
        if current_user.role != required_role and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required. You have '{current_user.role}'"
            )
        return current_user
    return role_checker