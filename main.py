import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import time

DATAFRAME_FILE = 'unique_cards.pkl'
INDEX_FILE = 'cards_faiss.index'

print("Cargando modelo de lenguaje...")
model = SentenceTransformer('all-MiniLM-L6-v2')

if os.path.exists(DATAFRAME_FILE) and os.path.exists(INDEX_FILE):
    print("Archivos pre-procesados encontrados. Cargando desde el disco...")
    start_time = time.time()
    df_unique = pd.read_pickle(DATAFRAME_FILE)
    index = faiss.read_index(INDEX_FILE)
    end_time = time.time()
    print(f"¡Carga completada en {end_time - start_time:.2f} segundos!")
else:
    print("Archivos no encontrados. Realizando el procesamiento completo por primera vez...")
    json_path = 'AllPrintings.json'
    all_cards_list = []
    print("Cargando y aplanando AllPrintings.json...")
    with open(json_path, 'r', encoding='utf-8') as f:
        all_sets = json.load(f)
    for set_code, set_data in all_sets['data'].items():
        for card_data in set_data['cards']:
            all_cards_list.append(card_data)
    df = pd.DataFrame(all_cards_list)
    print(f"Reduciendo {len(df)} impresiones a cartas únicas...")
    df_unique = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)
    df_unique['text'] = df_unique['text'].fillna('')
    print(f"Procesando {len(df_unique)} cartas únicas.")
    print("Generando embeddings (esto puede tardar varios minutos)...")
    card_texts = df_unique['text'].tolist()
    card_embeddings = model.encode(card_texts, show_progress_bar=True)
    print("Creando índice Faiss...")
    index = faiss.IndexFlatL2(card_embeddings.shape[1])
    index.add(card_embeddings)
    print("Guardando DataFrame y el índice Faiss en el disco...")
    df_unique.to_pickle(DATAFRAME_FILE)
    faiss.write_index(index, INDEX_FILE)
    print("¡Procesamiento inicial completado y archivos guardados!")

# while True:
#     query = input("\nIntroduce tu búsqueda (o escribe 'salir' para terminar): ")
#     if query.lower() == 'salir':
#         break
#     k = 10
#     query_embedding = model.encode([query])
#     distances, indices = index.search(query_embedding, k)
#     print(f"\nResultados para la consulta: '{query}'")
#     for i, dist in zip(indices[0], distances[0]):
#         similarity_score = 1 / (1 + dist)
#         card_name = df_unique.iloc[i]['name']
#         card_text = df_unique.iloc[i]['text'].replace('\n', ' ')
#         print(f"- Puntuación: {similarity_score:.2f} | {card_name}: '{card_text}'")


def hybrid_search(query, k=10, alpha=0.7):
    """
    Búsqueda híbrida por descripción:
    - alpha = peso de la parte semántica
    - (1-alpha) = peso de la coincidencia textual exacta
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k*3)  # pedimos más para enriquecer
    semantic_results = {}
    for i, dist in zip(indices[0], distances[0]):
        semantic_score = 1 / (1 + dist)
        semantic_results[i] = semantic_score
    
    textual_results = {}
    for i, row in df_unique.iterrows():
        if query.lower() in row['text'].lower():
            textual_results[i] = 1.0  # coincidencia exacta en descripción
    
    combined = {}
    all_ids = set(semantic_results.keys()) | set(textual_results.keys())
    for i in all_ids:
        sem_score = semantic_results.get(i, 0)
        txt_score = textual_results.get(i, 0)
        combined[i] = alpha * sem_score + (1 - alpha) * txt_score
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return sorted_results

while True:
    query = input("\nIntroduce tu búsqueda (o escribe 'salir' para terminar): ")
    if query.lower() == 'salir':
        break
    
    results = hybrid_search(query, k=10, alpha=0.7)
    
    print(f"\nResultados híbridos para la consulta: '{query}'")
    for i, score in results:
        card_name = df_unique.iloc[i]['name']
        card_text = df_unique.iloc[i]['text'].replace('\n', ' ')
        print(f"- Score: {score:.2f} | {card_name}: '{card_text}'")
