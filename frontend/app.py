import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

# ==================== LOGGING SETUP ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_error(error_msg: str, context: Dict = None, request_id: str = None):
    log_entry = {
        "error": error_msg,
        "context": context or {},
        "request_id": request_id or str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }
    logger.error(json.dumps(log_entry))
    st.error(f"❌ {error_msg[:200]}")

# ==================== DATA MODELS ====================

@dataclass
class MetricDefinition:
    key: str
    label: str
    value: str
    icon: str
    format_type: str

@dataclass
class ChartDefinition:
    type: str
    x_column: str
    y_column: str
    title: str
    format_type: Optional[str] = None

@dataclass
class FormattingRule:
    column: str
    format_type: str
    precision: int = 2

# ==================== STATE MANAGEMENT ====================

class AppState:
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        if "token" not in st.session_state:
            st.session_state.token = None
        if "user_email" not in st.session_state:
            st.session_state.user_email = None
        if "user_role" not in st.session_state:
            st.session_state.user_role = None
        if "question" not in st.session_state:
            st.session_state.question = ""
        if "page" not in st.session_state:
            st.session_state.page = "chat"
    
    @property
    def token(self):
        return st.session_state.token
    
    @token.setter
    def token(self, value):
        st.session_state.token = value
    
    @property
    def user_email(self):
        return st.session_state.user_email
    
    @user_email.setter
    def user_email(self, value):
        st.session_state.user_email = value
    
    @property
    def user_role(self):
        return st.session_state.user_role
    
    @user_role.setter
    def user_role(self, value):
        st.session_state.user_role = value
    
    @property
    def question(self):
        return st.session_state.question
    
    @question.setter
    def question(self, value):
        st.session_state.question = value
    
    @property
    def page(self):
        return st.session_state.page
    
    @page.setter
    def page(self, value):
        st.session_state.page = value
    
    def logout(self):
        st.session_state.token = None
        st.session_state.user_email = None
        st.session_state.user_role = None
        st.session_state.question = ""
        st.session_state.page = "chat"
        st.rerun()

# ==================== API CLIENT ====================

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def login(self, email: str, password: str) -> Dict:
        url = f"{self.base_url}/login"
        response = requests.post(
            url,
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def ask_question(self, question: str, token: str) -> Dict:
        url = f"{self.base_url}/ask"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            url,
            json={"question": question},
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    
    def get_monitoring_stats(self, token: str) -> Dict:
        url = f"{self.base_url}/monitoring/stats"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

# ==================== RENDERER ====================

class Renderer:
    @staticmethod
    def render_metrics(metrics: List[Dict]) -> None:
        if not metrics:
            return
        cols = st.columns(min(len(metrics), 4))
        for idx, metric in enumerate(metrics[:4]):
            with cols[idx]:
                st.metric(label=metric.get("label", ""), value=metric.get("value", ""))
    
    @staticmethod
    def render_chart(df: pd.DataFrame, chart: Dict) -> None:
        if df.empty or not chart:
            return
        if chart.get("type") == "bar":
            fig = px.bar(
                df, 
                x=chart.get("x_column"), 
                y=chart.get("y_column"),
                title=chart.get("title")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_sql(sql_query: str, explanation: str) -> None:
        if not sql_query:
            return
        with st.expander("🔍 View SQL Query"):
            st.code(sql_query, language="sql")
            if explanation:
                st.markdown(explanation)
    
    @staticmethod
    def render_table(df: pd.DataFrame) -> None:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "data.csv", "text/csv")

# ==================== CONFIGURATION ====================

st.set_page_config(page_title="AI Data Analyst", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp footer {display: none;}
        .stDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# FIXED: Using the correct backend URL
state = AppState()
api_client = APIClient("https://ai-analyst-copilot-2.onrender.com")

# ==================== LOGIN PAGE ====================
def show_login():
    st.title("🔐 AI Data Analyst Copilot")
    st.markdown("### Please Login to Continue")
    
    email = st.text_input("Email", placeholder="admin@company.com")
    password = st.text_input("Password", type="password", placeholder="admin123")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Login", type="primary", use_container_width=True):
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                try:
                    data = api_client.login(email, password)
                    state.token = data["access_token"]
                    state.user_email = data["user_email"]
                    state.user_role = data["role"]
                    st.success(f"✅ Welcome {email}!")
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend")
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 401:
                        st.error("❌ Invalid email or password")
                    else:
                        st.error(f"❌ Login failed: {e.response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:100]}")
    
    with st.expander("ℹ️ Demo Credentials"):
        st.markdown("""
        | Email | Password | Role |
        |-------|----------|------|
        | admin@company.com | admin123 | Admin |
        | analyst@company.com | analyst123 | Analyst |
        | viewer@company.com | viewer123 | Viewer |
        """)

# ==================== MONITORING PAGE ====================
def show_monitoring():
    st.title("📊 System Monitoring Dashboard")
    
    if st.button("← Back to Chat", use_container_width=True):
        state.page = "chat"
        st.rerun()
        return
    
    try:
        stats = api_client.get_monitoring_stats(state.token)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", stats.get("total_requests", 0))
        with col2:
            st.metric("Avg Response Time", f"{stats.get('avg_response_time_ms', 0)} ms")
        with col3:
            st.metric("Error Rate", f"{stats.get('error_rate', 0)}%")
        with col4:
            st.metric("Status", stats.get("status", "unknown").upper())
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")

# ==================== MAIN APP ====================
def show_main_app():
    with st.sidebar:
        st.markdown(f"### 👤 {state.user_email}")
        st.markdown(f"**Role:** `{state.user_role}`")
        st.markdown("---")
        
        if st.button("💬 Chat", use_container_width=True):
            state.page = "chat"
            st.rerun()
        
        if st.button("📊 Monitoring", use_container_width=True):
            state.page = "monitoring"
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            state.logout()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        for q in ["Show me all sales", "What is total revenue?", "Show sales by product"]:
            if st.button(q, key=q, use_container_width=True):
                state.question = q
                state.page = "chat"
                st.rerun()
        
        st.markdown("---")
        st.info("✅ PostgreSQL Connected\n✅ Gemini AI Ready")

    if state.page == "monitoring":
        show_monitoring()
        return
    
    st.title("🤖 AI Data Analyst Copilot")
    st.markdown("*Ask questions about your data in plain English*")

    question = st.text_area(
        "📝 **Ask your question:**", 
        value=state.question or "",
        height=100,
        placeholder="Example: Show me sales by product..."
    )

    if st.button("🔍 Ask", type="primary") and question:
        with st.spinner("Analyzing..."):
            try:
                data = api_client.ask_question(question, state.token)
                
                st.markdown("---")
                st.markdown("## 💡 Answer")
                st.success(data.get("answer", "No answer"))
                
                if data.get("metadata", {}).get("query_time_ms"):
                    st.caption(f"⚡ {data['metadata']['query_time_ms']} ms")
                
                if data.get("sql_used"):
                    with st.expander("🔍 View SQL Query"):
                        st.code(data["sql_used"], language="sql")
                
                if data.get("data") and len(data["data"]) > 0:
                    df = pd.DataFrame(data["data"])
                    
                    if data.get("metrics"):
                        Renderer.render_metrics(data["metrics"])
                    
                    tab1, tab2 = st.tabs(["📊 Data Table", "📈 Chart"])
                    with tab1:
                        Renderer.render_table(df)
                    with tab2:
                        Renderer.render_chart(df, data.get("chart"))
                            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Session expired. Please login again.")
                    state.token = None
                    st.rerun()
                else:
                    st.error(f"Error: {str(e)[:100]}")
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

# ==================== ENTRY POINT ====================
if state.token is None:
    show_login()
else:
    show_main_app()
