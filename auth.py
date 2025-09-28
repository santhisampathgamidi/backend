# auth.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt

# ------------------------------
# Config
# ------------------------------
SECRET_KEY = "supersecretkey"  # 🔑 Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

router = APIRouter()
security = HTTPBasic()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

# Mock users (username → password, role)
USERS = {
    "leasee_user": {"password": "password123", "role": "leasee"},
    "lessor_user": {"password": "password456", "role": "lessor"},
}

# ------------------------------
# Models
# ------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    role: str

# ------------------------------
# Helpers
# ------------------------------
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ------------------------------
# Routes
# ------------------------------
@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest):
    user = USERS.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token_data = {"sub": request.username, "role": user["role"]}
    token = create_access_token(token_data)

    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

# ------------------------------
# Dependency for protected routes
# ------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    username = payload.get("sub")
    role = payload.get("role")

    if username is None or role is None:
        raise HTTPException(status_code=401, detail="Invalid authentication")

    return {"username": username, "role": role}
