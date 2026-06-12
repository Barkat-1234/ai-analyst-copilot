import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Data Analyst", 
    layout="wide",
    page_icon="🤖"
)

# Custom CSS
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp footer {display: none;}
        .stDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
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

# IMPORTANT: Change this to your Render backend URL
API_URL = "https://ai-analyst-copilot-2.onrender.com"

# ==================== LOGIN PAGE ====================
def show_login():
    st.title("🔐 AI Data Analyst Copilot")
    st.markdown("### Please Login to Continue")
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="admin@company.com")
        password = st.text_input("Password", type="password", placeholder="admin123")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
        
        if submitted:
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/login",
                        json={"email": email, "password": password},
                        timeout=30
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.token = data["access_token"]
                        st.session_state.user_email = data["user_email"]
                        st.session_state.user_role = data["role"]
                        st.success(f"✅ Welcome {email}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to backend at {API_URL}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
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
    st.markdown("*Real-time system statistics*")
    
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{API_URL}/monitoring/stats", headers=headers, timeout=30)
        
        if response.status_code == 200:
            stats = response.json()
            
            # Display metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", stats.get("total_requests", 0))
            with col2:
                st.metric("Avg Response Time", f"{stats.get('avg_response_time_ms', 0)} ms")
            with col3:
                st.metric("Error Rate", f"{stats.get('error_rate', 0)}%")
            with col4:
                st.metric("Status", stats.get("status", "unknown").upper())
            
            # Top questions
            if stats.get("top_questions"):
                st.markdown("---")
                st.subheader("🔥 Most Asked Questions")
                top_df = pd.DataFrame(stats["top_questions"], columns=["Question", "Count"])
                st.dataframe(top_df, use_container_width=True)
            
            # System health
            st.markdown("---")
            st.subheader("🩺 System Health")
            if stats.get("status") == "healthy":
                st.success("All systems operational")
            else:
                st.warning("System issues detected")
                
        elif response.status_code == 403:
            st.error("❌ You don't have permission to view monitoring (Admin only)")
        else:
            st.error(f"Error: {response.status_code}")
            
    except Exception as e:
        st.error(f"Error fetching monitoring data: {e}")
    
    if st.button("← Back to Chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()

# ==================== MAIN APP ====================
def show_main_app():
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.user_email}")
        st.markdown(f"**Role:** `{st.session_state.user_role}`")
        st.markdown("---")
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
        
        if st.button("📊 Monitoring", use_container_width=True):
            st.session_state.page = "monitoring"
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["token", "user_email", "user_role", "question", "page"]:
                st.session_state[key] = None if key != "page" else "chat"
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        example_questions = [
            "Show me all sales",
            "What is total revenue?",
            "Show sales by region",
            "Which product sold the most?",
            "Total sales by product"
        ]
        for q in example_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.question = q
                st.session_state.page = "chat"
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Database Info")
        st.info("✅ PostgreSQL Connected")
        st.info("✅ Gemini AI Ready")
        st.info("✅ RAG System Active")

    # Page routing
    if st.session_state.get("page") == "monitoring":
        show_monitoring()
        return
    
    # Chat page
    st.title("🤖 AI Data Analyst Copilot")
    st.markdown("*Ask questions about your data in plain English*")

    question = st.text_area(
        "📝 **Ask your question:**", 
        value=st.session_state.question if st.session_state.question else "",
        height=100,
        placeholder="Example: Show me all sales by region..."
    )

    if st.button("🔍 Ask", type="primary"):
        if question:
            with st.spinner("🧠 Analyzing..."):
                try:
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    response = requests.post(
                        f"{API_URL}/ask",
                        json={"question": question},
                        headers=headers,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.markdown("---")
                        st.markdown("## 💡 Answer")
                        st.success(data["answer"])
                        
                        # Show query time in metadata
                        if data.get("metadata"):
                            query_time = data["metadata"].get("query_time_ms", 0)
                            st.caption(f"⚡ Query completed in {query_time} ms")
                        
                        if data.get("data") and len(data["data"]) > 0:
                            df = pd.DataFrame(data["data"])
                            
                            tab1, tab2 = st.tabs(["📊 Data Table", "📈 Bar Chart"])
                            
                            with tab1:
                                st.dataframe(df, use_container_width=True)
                                csv = df.to_csv(index=False)
                                st.download_button("📥 Download CSV", csv, "data.csv", "text/csv")
                            
                            with tab2:
                                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                                if numeric_cols:
                                    y_col = numeric_cols[0]
                                    x_col = df.columns[0]
                                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                    st.plotly_chart(fig, use_container_width=True)
                    elif response.status_code == 401:
                        st.error("Session expired. Please login again.")
                        st.session_state.token = None
                        st.rerun()
                    else:
                        st.error(f"Error: {response.status_code}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try again.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question")

# ==================== MAIN ====================
if st.session_state.token is None:
    show_login()
else:
    show_main_app()
