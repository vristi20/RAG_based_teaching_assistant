import streamlit as st
import os
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DB_FILENAME = "vector_database.joblib"

# --- Page Config & Theme Styling ---
st.set_page_config(page_title="EduNexus AI Workspace", page_icon="🧠", layout="wide")

# Custom Strong-Override Nordic Slate CSS Styling
st.markdown("""
    <style>
    div[data-testid="stAppViewContainer"], .stApp {
        background-color: #1a222d !important;
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] {
        background-color: #121820 !important;
        border-right: 1px solid #2d3748 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
    }
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

# --- Initialize Cloud-safe Models ---
@st.cache_resource
def load_vector_db():
    if os.path.exists(DB_FILENAME):
        return joblib.load(DB_FILENAME)
    return None

@st.cache_resource
def load_embedding_model():
    # Downloads and runs the encoder directly inside the cloud server container memory
    return SentenceTransformer("BAAI/bge-m3")

# Setup Groq Client using Streamlit Secrets Management
if "GROQ_API_KEY" in st.secrets:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
else:
    st.error("Missing Groq API Key. Please configure GROQ_API_KEY in Streamlit Secrets.")
    st.stop()

# Load data assets safely
df = load_vector_db()
embedding_model = load_embedding_model()

# --- SIDEBAR (Control Panel & Metrics) ---
with st.sidebar:
    st.markdown("### 🛠️ Engine Diagnostics")
    st.markdown("---")
    if df is not None:
        st.success("Knowledge Index Connected")
        st.metric(label="Indexed Text Segments", value=f"{len(df):,}")
        st.metric(label="Retrieval Layer", value="bge-m3 Cloud Vector")
        st.metric(label="Synthesis Model", value="Llama3 (via Groq Cloud)")
    else:
        st.error("Database Offline - Vector index missing from repository root.")
        st.stop()
        
    st.markdown("---")
    if st.button("🧹 Clear Conversation Record", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN DASHBOARD LAYOUT ---
st.markdown('<div class="main-title">🧠 EduNexus Knowledge Explorer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Query, locate, and cross-reference context across your video training library effortlessly.</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
with m1:
    st.info("⚡ **Memory Cache:** Pre-loaded lookup matrix active.")
with m2:
    st.info("🎯 **Similarity Search:** Dense vector structural cosine comparisons engaged.")
with m3:
    st.info("🛡️ **Cloud Infrastructure:** High-speed inference optimization running via Groq.")

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_column, reference_column = st.columns([1.25, 0.75], gap="large")

with chat_column:
    st.subheader("💬 Workspace Chat")
    
    chat_container = st.container(height=480, border=True)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("<p style='color:#94a3b8; text-align:center; padding-top:120px; font-style: italic;'>Submit a course query below to initialize local document retrieval synthesis.</p>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_question := st.chat_input("Ask a technical question about your courses..."):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with chat_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                
                # Generate tracking vector natively in cloud container
                question_embedding = embedding_model.encode([user_question])[0]
                
                if question_embedding is not None:
                    similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
                    top_indices = similarities.argsort()[::-1][:3]
                    matched_df = df.loc[top_indices]
                    
                    st.session_state.last_context = []
                    context_text = ""
                    for idx, row in matched_df.iterrows():
                        context_text += f"\n{row['text']}\n"
                        st.session_state.last_context.append({
                            "title": row['video_title'],
                            "chunk": row.get('chunk_id', 'N/A'),
                            "text": row['text']
                        })
                        
                    rag_prompt = f"""You are a helpful video course assistant. Answer the user's question accurately using ONLY the provided text context below.
Context:
{context_text}

User Question: {user_question}
Answer:"""
                    
                    # Call out to the cloud inference layer
                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[{"role": "user", "content": rag_prompt}],
                        stream=True,
                    )
                    
                    full_response = ""
                    for chunk in completion:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
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