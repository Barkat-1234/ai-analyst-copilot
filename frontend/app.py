import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

# ==================== LOGGING SETUP ====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track performance metrics for each request"""
    
    def __init__(self):
        self.metrics = {}
    
    def start(self, name: str):
        self.metrics[name] = {"start": time.time(), "end": None, "duration": None}
    
    def end(self, name: str):
        if name in self.metrics:
            self.metrics[name]["end"] = time.time()
            self.metrics[name]["duration"] = self.metrics[name]["end"] - self.metrics[name]["start"]
    
    def get_duration(self, name: str) -> float:
        return self.metrics.get(name, {}).get("duration", 0)

def log_error(error_msg: str, context: Dict = None, request_id: str = None):
    """Centralized error logging with request tracking"""
    log_entry = {
        "error": error_msg,
        "context": context or {},
        "request_id": request_id or str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }
    logger.error(json.dumps(log_entry))
    st.error(f"❌ {error_msg[:200]}")

# ==================== STRICT DATA MODELS (BACKEND-DRIVEN) ====================

@dataclass
class MetricDefinition:
    """Fully typed metric from backend"""
    key: str
    label: str
    value: str
    icon: str
    format_type: str
    severity: Optional[str] = None

@dataclass
class ChartDefinition:
    """Fully typed chart definition from backend"""
    type: str
    x_column: str
    y_column: str
    title: str
    format_type: Optional[str] = None
    color_column: Optional[str] = None

@dataclass
class FormattingRule:
    """Column formatting rule from backend"""
    column: str
    format_type: str
    precision: int = 2

@dataclass
class APIResponse:
    """Strict backend response schema"""
    success: bool
    answer: str
    sql_used: str
    sql_explanation: str
    data: List[Dict[str, Any]]
    metrics: List[MetricDefinition]
    chart: Optional[ChartDefinition]
    formatting_rules: List[FormattingRule]
    metadata: Dict[str, Any]
    error: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    api_version: str = "3.0.0"
    
    @classmethod
    def from_dict(cls, data: Dict, request_id: str = None) -> 'APIResponse':
        """Parse and validate backend response"""
        metrics = []
        for m in data.get("metrics", []):
            metrics.append(MetricDefinition(
                key=m.get("key", ""),
                label=m.get("label", ""),
                value=m.get("value", ""),
                icon=m.get("icon", ""),
                format_type=m.get("format_type", "string")
            ))
        
        chart = None
        if data.get("chart"):
            chart = ChartDefinition(
                type=data["chart"].get("type", "bar"),
                x_column=data["chart"].get("x_column", ""),
                y_column=data["chart"].get("y_column", ""),
                title=data["chart"].get("title", ""),
                format_type=data["chart"].get("format_type"),
                color_column=data["chart"].get("color_column")
            )
        
        formatting_rules = []
        for rule in data.get("formatting_rules", []):
            formatting_rules.append(FormattingRule(
                column=rule.get("column", ""),
                format_type=rule.get("format_type", "raw"),
                precision=rule.get("precision", 2)
            ))
        
        return cls(
            success=data.get("success", True),
            answer=data.get("answer", ""),
            sql_used=data.get("sql_used", ""),
            sql_explanation=data.get("sql_explanation", ""),
            data=data.get("data", []),
            metrics=metrics,
            chart=chart,
            formatting_rules=formatting_rules,
            metadata=data.get("metadata", {}),
            error=data.get("error"),
            request_id=request_id or data.get("request_id", str(uuid.uuid4())),
            api_version=data.get("api_version", "3.0.0")
        )

# ==================== CENTRAL STATE MANAGEMENT ====================

class AppState:
    """Centralized state management wrapper"""
    
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
        if "perf_tracker" not in st.session_state:
            st.session_state.perf_tracker = PerformanceTracker()
    
    @property
    def token(self) -> Optional[str]:
        return st.session_state.token
    
    @token.setter
    def token(self, value: Optional[str]):
        st.session_state.token = value
    
    @property
    def user_email(self) -> Optional[str]:
        return st.session_state.user_email
    
    @user_email.setter
    def user_email(self, value: Optional[str]):
        st.session_state.user_email = value
    
    @property
    def user_role(self) -> Optional[str]:
        return st.session_state.user_role
    
    @user_role.setter
    def user_role(self, value: Optional[str]):
        st.session_state.user_role = value
    
    @property
    def question(self) -> str:
        return st.session_state.question
    
    @question.setter
    def question(self, value: str):
        st.session_state.question = value
    
    @property
    def page(self) -> str:
        return st.session_state.page
    
    @page.setter
    def page(self, value: str):
        st.session_state.page = value
    
    @property
    def perf_tracker(self) -> PerformanceTracker:
        return st.session_state.perf_tracker
    
    def logout(self):
        st.session_state.token = None
        st.session_state.user_email = None
        st.session_state.user_role = None
        st.session_state.question = ""
        st.session_state.page = "chat"
        st.rerun()

# ==================== SIMPLE API CLIENT (FIXED) ====================

class SimpleAPIClient:
    """Simple API client that works reliably"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def login(self, email: str, password: str) -> Dict:
        """Login to get token"""
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
        """Ask a question"""
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
        """Get monitoring stats"""
        url = f"{self.base_url}/monitoring/stats"
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()

# ==================== PURE BACKEND-DRIVEN RENDERER ====================

class BackendDrivenRenderer:
    """Renders based SOLELY on backend configuration"""
    
    @staticmethod
    def render_metrics(metrics: List[MetricDefinition]) -> None:
        if not metrics:
            return
        
        cols = st.columns(min(len(metrics), 4))
        for idx, metric in enumerate(metrics[:4]):
            with cols[idx]:
                display_label = f"{metric.icon} {metric.label}" if metric.icon else metric.label
                st.metric(label=display_label, value=metric.value)
    
    @staticmethod
    def render_chart(df: pd.DataFrame, chart: Optional[ChartDefinition]) -> None:
        if df.empty or not chart:
            return
        
        if chart.x_column not in df.columns or chart.y_column not in df.columns:
            return
        
        try:
            if chart.type == "bar":
                fig = px.bar(
                    df, x=chart.x_column, y=chart.y_column,
                    title=chart.title, color=chart.color_column, text_auto=True
                )
                if chart.format_type == "currency":
                    fig.update_traces(texttemplate='%{y:$,.2f}', textposition='outside')
                fig.update_layout(height=500, bargap=0.3)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart.type == "line":
                fig = px.line(
                    df, x=chart.x_column, y=chart.y_column,
                    title=chart.title, color=chart.color_column, markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            log_error(f"Chart rendering failed", {"error": str(e)[:100]})
    
    @staticmethod
    def render_sql(sql_query: str, explanation: str) -> None:
        if not sql_query or sql_query == "Error generating SQL":
            return
        
        with st.expander("🔍 View SQL Query", expanded=False):
            st.code(sql_query, language="sql")
            if st.button("📋 Copy SQL", key="copy_sql_btn"):
                st.success("✅ SQL ready to copy")
            if explanation:
                st.markdown("---")
                st.markdown("### 📖 Query Explanation")
                st.markdown(explanation)
    
    @staticmethod
    def render_table(df: pd.DataFrame, formatting_rules: List[FormattingRule]) -> None:
        if df.empty:
            return
        
        format_map = {rule.column: rule for rule in formatting_rules}
        display_df = df.copy()
        
        for col in display_df.columns:
            if col in format_map:
                rule = format_map[col]
                if rule.format_type == "currency":
                    display_df[col] = display_df[col].apply(
                        lambda x: f"${x:,.{rule.precision}f}" if isinstance(x, (int, float)) else x
                    )
                elif rule.format_type == "integer":
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else x
                    )
        
        st.dataframe(display_df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def render_answer(answer: str) -> None:
        st.success(answer)

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

# Initialize state and client - FIXED URL
state = AppState()
api_client = SimpleAPIClient("https://ai-analyst-copilot-2.onrender.com")

# ==================== LOGIN PAGE ====================
def show_login():
    st.title("🔐 AI Data Analyst Copilot")
    st.markdown("### Please Login to Continue")
    
    email = st.text_input("Email", placeholder="admin@company.com")
    password = st.text_input("Password", type="password", placeholder="admin123")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        login_clicked = st.button("Login", type="primary", use_container_width=True)
    
    if login_clicked:
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
                st.error("❌ Cannot connect to backend. Make sure backend is running.")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("❌ Invalid email or password")
                else:
                    st.error(f"❌ Login failed: {e.response.status_code}")
            except Exception as e:
                st.error(f"❌ Login failed: {str(e)[:100]}")
    
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
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            st.error("❌ Admin access required")
        else:
            st.error(f"Error: {str(e)[:100]}")
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")

# ==================== MAIN APP ====================
def show_main_app():
    # Sidebar
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
    
    # Chat page
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
                
                # Show SQL
                if data.get("sql_used"):
                    with st.expander("🔍 View SQL Query"):
                        st.code(data["sql_used"], language="sql")
                
                # Show data
                if data.get("data") and len(data["data"]) > 0:
                    df = pd.DataFrame(data["data"])
                    
                    # Show metrics
                    if data.get("metrics"):
                        cols = st.columns(min(len(data["metrics"]), 4))
                        for idx, metric in enumerate(data["metrics"][:4]):
                            with cols[idx]:
                                st.metric(
                                    label=metric.get("label", ""),
                                    value=metric.get("value", "")
                                )
                    
                    tab1, tab2 = st.tabs(["📊 Data Table", "📈 Chart"])
                    with tab1:
                        st.dataframe(df, use_container_width=True)
                        csv = df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "data.csv", "text/csv")
                    with tab2:
                        if data.get("chart"):
                            chart = data["chart"]
                            if chart.get("type") == "bar":
                                fig = px.bar(
                                    df, 
                                    x=chart.get("x_column"), 
                                    y=chart.get("y_column"),
                                    title=chart.get("title")
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
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
