import requests
import os
import json
import pandas as pd
import numpy as np
import joblib  # For saving/loading the binary database efficiently
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def create_embedding(text):
    """
    Sends a single text chunk to Ollama.
    Processes one chunk at a time per thread to prevent connection refusal crashes.
    """
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": [text]
        }, timeout=30)
        
        response_data = r.json()
        if "embeddings" in response_data:
            return response_data["embeddings"][0]
        else:
            print(f"Ollama error: {response_data.get('error')}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

# --- Main Logic ---
if __name__ == "__main__":
    db_filename = "vector_database.joblib"
    
    # 1. Check if we already have the database saved to skip processing entirely!
    if os.path.exists(db_filename):
        print(f"Found existing database '{db_filename}'. Loading it directly...")
        df = joblib.load(db_filename)
        print(f"Successfully loaded {len(df)} chunks. Ready for search!")
    else:
        # If no database exists, process everything from scratch
        jsons_dir = "jsons"
        if not os.path.exists(jsons_dir):
            print(f"Error: The folder '{jsons_dir}' was not found.")
            exit()

        jsons = os.listdir(jsons_dir)
        my_dicts = []
        all_chunks_to_process = []

        print("Reading JSON files...")
        for json_file in jsons:
            if not json_file.endswith('.json'):
                continue
                
            with open(os.path.join(jsons_dir, json_file), 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            video_title = json_file.replace('.json', '')
            
            for chunk in content.get('chunks', []):
                chunk['video_title'] = video_title
                all_chunks_to_process.append(chunk)

        total_chunks = len(all_chunks_to_process)
        print(f"Total chunks found across all files: {total_chunks}")
        
        if total_chunks == 0:
            print("No chunks found to process.")
            exit()

        print("\nGenerating embeddings concurrently... (Terminal optimized)")

        # 2. Process chunks concurrently using 4 parallel workers
        max_workers = 4 
        just_texts = [item['text'] for item in all_chunks_to_process]
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(create_embedding, text) for text in just_texts]
            
            # miniters=100 keeps VS Code terminal from choking on progress bar updates
            for future in tqdm(futures, total=len(futures), desc="Processing Chunks", miniters=100, maxinterval=5.0):
                results.append(future.result())

        # 3. Reassemble everything
        chunk_id = 0
        for chunk, embedding in zip(all_chunks_to_process, results):
            if embedding is not None:
                chunk['chunk_id'] = chunk_id
                chunk['embedding'] = embedding
                my_dicts.append(chunk)
                chunk_id += 1

        df = pd.DataFrame(my_dicts)
        
        print("\n--- Processing Complete ---")
        print(f"Total Chunks Successfully Processed: {len(df)}")
        
        # 4. Save your vector database using Joblib with zip compression
        joblib.dump(df, db_filename, compress=3)  
        print(f"Saved complete binary vector database locally to: {db_filename}")

    # 5. --- RAG QUERY SECTION ---
    print("\n" + "="*50)
    print("RAG Query System Initialized")
    print("="*50)
    
    incoming_query = input("Ask a Question: ")
    question_embedding = create_embedding(incoming_query)

    if question_embedding is not None:
        # Calculate similarities 
        similarities = cosine_similarity(np.vstack(df['embedding'].values), [question_embedding]).flatten()
        
        # Pull top 3 matches
        top_results = 3
        max_indx = similarities.argsort()[::-1][0:top_results]
        
        new_df = df.loc[max_indx] 
        print("\n--- Top Relevant Context Found ---")
        
        # Loop explicitly to cleanly print matching sections
        for idx, row in new_df.iterrows():
            print(f"\n[Source Video]: {row['video_title']} (Chunk: {row['chunk_id']})")
            print(f"[Context]: {row['text']}\n" + "-"*40)
    else:
        print("Failed to generate an embedding for your query.")