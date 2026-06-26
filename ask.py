import os
import requests
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define which LLM model you want to use for answering
LLM_MODEL = "llama3"  # Change this to whatever model you have pulled (e.g., "mistral", "gemma")

def get_query_embedding(text):
    """Turns your new question into a vector using Ollama so we can compare it."""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": [text]
        }, timeout=10)
        
        response_data = r.json()
        if "embeddings" in response_data:
            return response_data["embeddings"][0]
        else:
            print(f"Ollama embedding error: {response_data.get('error')}")
            return None
    except Exception as e:
        print(f"Failed to connect to Ollama for embedding: {e}")
        return None

def generate_llm_answer(prompt):
    """Sends the complete structured prompt to the LLM and streams the answer."""
    try:
        # Using stream=True allows us to print the words as they are generated live
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True
        }, stream=True, timeout=60)
        
        if r.status_code != 200:
            print(f"Ollama generation error code: {r.status_code}")
            return
            
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                response_text = chunk.get("response", "")
                print(response_text, end="", flush=True)
        print() # New line at the very end
    except Exception as e:
        print(f"\nFailed to get generation response from LLM: {e}")

if __name__ == "__main__":
    import json # imported here to keep dependencies clean
    db_filename = "vector_database.joblib"
    
    if not os.path.exists(db_filename):
        print(f"Error: Vector database file '{db_filename}' was not found.")
        print("Please run your 'csv_joblib.py' conversion file first!")
        exit()
        
    print(f"Loading database from '{db_filename}'...")
    df = joblib.load(db_filename)
    print(f"Successfully loaded {len(df)} chunks into memory.")
    print("=" * 60)
    print(f"RAG Assistant Ready (Using LLM: {LLM_MODEL})")
    print("Type your question below. (Type 'quit' or 'exit' to stop)")
    print("=" * 60)
    
    while True:
        incoming_query = input("\nAsk a Question: ").strip()
        
        if incoming_query.lower() in ['quit', 'exit', '']:
            print("Closing search assistant. Goodbye!")
            break
            
        print("Searching database and formulating answer...")
        question_embedding = get_query_embedding(incoming_query)
        
        if question_embedding is not None:
            # Match vectors
            similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
            
            # Fetch top 3 closest context items
            top_results = 3
            max_indices = similarities.argsort()[::-1][:top_results]
            new_df = df.loc[max_indices]
            
            # Combine the texts from our top matches to create our context block
            context_text = ""
            for idx, row in new_df.iterrows():
                context_text += f"\n--- Source: {row['video_title']} ---\n{row['text']}\n"
            
            # --- THE RAG PROMPT TEMPLATE ---
            rag_prompt = f"""You are a helpful video course assistant. Answer the user's question accurately using ONLY the provided text context below. 
If the context does not contain the answer, politely state that you do not know based on the provided materials. Do not make things up.

Context:
{context_text}

User Question: {incoming_query}

Answer:"""
            
            print("\n" + "="*20 + " ASSISTANT ANSWER " + "="*20)
            generate_llm_answer(rag_prompt)
            print("=" * 58)
            
        else:
            print("Could not process question. Please check that Ollama is running.")