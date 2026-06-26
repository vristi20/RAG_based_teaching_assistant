# 📚 EduNexus Knowledge Explorer

An AI-powered **Retrieval-Augmented Generation (RAG)** application that enables users to upload educational videos and interact with their content through natural language. The application extracts audio from videos, transcribes speech using Whisper, generates semantic embeddings, and retrieves the most relevant information to answer user queries using locally hosted Large Language Models (LLMs). Built with **Streamlit**, EduNexus provides an intuitive interface for interactive learning while ensuring complete offline processing through **Ollama**.

---

## 🌐 Live Demo

🚀 **Try the application here:**  
https://ragbasedteachingassistant-6q6gmyzur3bbh9a6gi7dae.streamlit.app/

---

## 🚀 Features

- 📹 Upload educational video files
- 🎧 Extract audio using FFmpeg
- 📝 Convert speech to text with OpenAI Whisper
- ✂️ Automatic transcript chunking
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered question answering with Llama 3 or Mistral
- 💬 Context-aware Retrieval-Augmented Generation (RAG)
- 💻 Interactive Streamlit web interface
- 🔒 Fully offline inference using Ollama

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend Development |
| Streamlit | Web Application |
| Ollama | Local LLM Hosting |
| Llama 3 / Mistral | Language Models |
| Whisper | Speech-to-Text Transcription |
| FFmpeg | Audio Extraction |
| LangChain | RAG Pipeline |
| ChromaDB | Vector Database |
| Sentence Transformers | Text Embeddings |

---

## 📂 Project Structure

```text
EduNexus-Knowledge-Explorer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── videos/
├── audio/
├── transcripts/
├── embeddings/
├── screenshots/
└── utils/
```

---

## ⚙️ How It Works

1. Upload an educational video.
2. Extract audio using FFmpeg.
3. Generate transcripts using Whisper.
4. Split transcripts into smaller chunks.
5. Convert text chunks into vector embeddings.
6. Store embeddings in ChromaDB.
7. Retrieve the most relevant chunks based on the user's query.
8. Generate accurate responses using a local LLM (Llama 3 or Mistral) via Ollama.

---

## 📸 Application Preview

Create a folder named **screenshots** and place your application images there.

```
screenshots/
├── home.png
├── upload.png
├── response.png
```

Then add them like this:

```markdown
### Home Page
![Home](screenshots/home.png)

### Upload Video
![Upload](screenshots/upload.png)

### Chat Interface
![Response](screenshots/response.png)
```

---

## ▶️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/vristi20/EduNexus-Knowledge-Explorer.git
cd EduNexus-Knowledge-Explorer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install FFmpeg

Download and install FFmpeg, then ensure it is added to your system PATH.

### 4. Install Ollama

Install Ollama and download a language model.

```bash
ollama pull llama3
```

or

```bash
ollama pull mistral
```

### 5. Run the Application

```bash
streamlit run app.py
```

---

## 📋 Requirements

- Python 3.10+
- FFmpeg
- Ollama
- Whisper
- Streamlit
- LangChain
- ChromaDB
- Sentence Transformers

---

## 🎯 Use Cases

- Interactive learning assistant
- Educational video question answering
- Lecture revision
- Course content exploration
- Offline AI-powered knowledge retrieval
- Student self-learning companion

---

## 🔮 Future Enhancements

- Support multiple videos in a single knowledge base
- PDF and document integration
- Chat history and conversation memory
- Source citations for responses
- User authentication
- Cloud deployment
- Multi-language support

---

## 👨‍💻 Author

**Vristi Sharma**

📧 Feel free to connect and explore my work!

**GitHub:** https://github.com/vristi20

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. Your support helps motivate future improvements and open-source contributions.
