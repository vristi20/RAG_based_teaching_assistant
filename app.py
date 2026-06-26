import streamlit as st
import os
import requests
import joblib
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
LLM_MODEL = "llama3"
DB_FILENAME = "vector_database.joblib"

# --- Page Config & Theme Styling ---
st.set_page_config(page_title="EduNexus AI Workspace", page_icon="🧠", layout="wide")

# Custom Strong-Override Nordic Slate CSS Styling
st.markdown("""
    <style>
    /* Force override Streamlit's native main container backdrop */
    div[data-testid="stAppViewContainer"], .stApp {
        background-color: #1a222d !important;
        color: #e2e8f0 !important;
    }
    
    /* Force override Streamlit's sidebar wrapper */
    div[data-testid="stSidebar"] {
        background-color: #121820 !important;
        border-right: 1px solid #2d3748 !important;
    }
    
    /* Smooth Metrics panel styling changes */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
    }
    
    /* Custom header styling with premium smooth gradient */
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    
    /* Refined soft source card designs */
    .context-card {
        background-color: #242f41 !important;
        border-left: 4px solid #818cf8 !important;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .context-source {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions Backend ---
@st.cache_resource
def load_vector_db():
    if os.path.exists(DB_FILENAME):
        return joblib.load(DB_FILENAME)
    return None

def get_query_embedding(text):
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": [text]
        }, timeout=10)
        response_data = r.json()
        if "embeddings" in response_data:
            return response_data["embeddings"][0]
    except Exception as e:
        st.error(f"Failed to connect to Ollama embedding service: {e}")
    return None

def stream_llm_response(prompt):
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True
        }, stream=True, timeout=60)
        
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                yield chunk.get("response", "")
    except Exception as e:
        yield f"\nError communicating with LLM: {e}"

# --- Data Engine Loading ---
df = load_vector_db()

# --- SIDEBAR (Control Panel & Metrics) ---
with st.sidebar:
    st.markdown("### 🛠️ Engine Diagnostics")
    st.markdown("---")
    if df is not None:
        st.success("Knowledge Index Connected")
        st.metric(label="Indexed Text Segments", value=f"{len(df):,}")
        st.metric(label="Retrieval Layer", value="bge-m3 Vector Embeddings")
        st.metric(label="Synthesis Model", value=LLM_MODEL.lower())
    else:
        st.error("Database Offline")
        st.stop()
        
    st.markdown("---")
    if st.button("🧹 Clear Conversation Record", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN DASHBOARD LAYOUT ---
st.markdown('<div class="main-title">🧠 EduNexus Knowledge Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Query, locate, and cross-reference context across your video training library effortlessly.</div>', unsafe_allow_html=True)

# Streamlined Performance Tracker Grid at the Top
m1, m2, m3 = st.columns(3)
with m1:
    st.info("⚡ **Memory Cache:** Pre-loaded `.joblib` lookup matrix is active.")
with m2:
    st.info("🎯 **Similarity Search:** Dense vector structural cosine comparisons are engaged.")
with m3:
    st.info("🛡️ **Data Security:** Full offline local compute isolation active via Ollama.")

st.markdown("---")

# Initialize Chat System
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout: Split Chat interface from Live Reference inspect module
chat_column, reference_column = st.columns([1.25, 0.75], gap="large")

with chat_column:
    st.subheader("💬 Workspace Chat")
    
    # Message Box Rendering
    chat_container = st.container(height=480, border=True)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("<p style='color:#94a3b8; text-align:center; padding-top:120px; font-style: italic;'>Submit a course query below to initialize local document retrieval synthesis.</p>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Text Input Logic Box
    if user_question := st.chat_input("Ask a technical question about your courses..."):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # Fetching contextual embeddings
                question_embedding = get_query_embedding(user_question)
                
                if question_embedding is not None:
                    similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
                    top_indices = similarities.argsort()[::-1][:3]
                    matched_df = df.loc[top_indices]
                    
                    # Store matches dynamically in session state to show on the side column card views
                    st.session_state.last_context = []
                    context_text = ""
                    for idx, row in matched_df.iterrows():
                        context_text += f"\n{row['text']}\n"
                        st.session_state.last_context.append({
                            "title": row['video_title'],
                            "chunk": row.get('chunk_id', 'N/A'),
                            "text": row['text']
                        })
                        
                    # Strict System Prompt Engineering Context
                    rag_prompt = f"""You are a helpful video course assistant. Answer the user's question accurately using ONLY the provided text context below.
Do not mention raw video paths or data schema structures in your final sentences. Fix syntax clarity issues gracefully.

Context:
{context_text}

User Question: {user_question}

Answer:"""
                    
                    # Process and stream generation text pipeline
                    full_response = ""
                    for chunk in stream_llm_response(rag_prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + " ▌")
                    response_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()

with reference_column:
    st.subheader("🔍 Grounding References")
    if "last_context" in st.session_state and st.session_state.last_context:
        for idx, src in enumerate(st.session_state.last_context):
            st.markdown(f"""
            <div class="context-card">
                <div class="context-source">📖 Source {idx+1}: {src['title']} (Segment {src['chunk']})</div>
                <div style="font-size:0.92rem; line-height:1.5; color:#cbd5e1;">"{src['text']}"</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#94a3b8; font-style:italic;'>The verified text block segments retrieved from your database to formulate your AI responses will dynamically appear here.</p>", unsafe_allow_html=True)