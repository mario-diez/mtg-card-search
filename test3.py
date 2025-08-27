# coding=utf-8
import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import time

DATAFRAME_FILE = 'unique_cards.pkl'
INDEX_FILE = 'cards_faiss.index'

print("Cargando modelos...")
bi_encoder = SentenceTransformer('all-mpnet-base-v2') 
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

if os.path.exists(DATAFRAME_FILE) and os.path.exists(INDEX_FILE):
    print("Archivos pre-procesados encontrados. Cargando desde el disco...")
    start_time = time.time()
    df_unique = pd.read_pickle(DATAFRAME_FILE)
    index = faiss.read_index(INDEX_FILE)
    end_time = time.time()
    print(f"¡Carga completada en {end_time - start_time:.2f} segundos!")
else:
    print("Archivos no encontrados. Procesando AllPrintings.json...")
    json_path = 'AllPrintings.json'
    all_cards_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        all_sets = json.load(f)
    for set_code, set_data in all_sets['data'].items():
        for card_data in set_data['cards']:
            all_cards_list.append(card_data)
    df = pd.DataFrame(all_cards_list)

    df_unique = df.drop_duplicates(subset=['name'], keep='first').reset_index(drop=True)
    df_unique['text'] = df_unique['text'].fillna('')
    print(df_unique.columns)
    df_unique['type'] = df_unique['type'].fillna('')

    df_unique["full_text"] = (
        df_unique["text"].fillna("") + " " +
        df_unique["type"].fillna("") + " " +
        df_unique["manaCost"].fillna("")
    )
    print(f"Procesando {len(df_unique)} cartas únicas...")
    print("Generando embeddings (esto puede tardar varios minutos)...")
    card_texts = df_unique['full_text'].tolist()
    card_embeddings = bi_encoder.encode(card_texts, show_progress_bar=True)

    print("Creando índice Faiss...")
    index = faiss.IndexFlatL2(card_embeddings.shape[1])
    index.add(card_embeddings)

    print("Guardando en disco...")
    df_unique.to_pickle(DATAFRAME_FILE)
    faiss.write_index(index, INDEX_FILE)
    print("¡Procesamiento inicial completado!")

def semantic_search(query, k=10, fetch_k=30):
    query_embedding = bi_encoder.encode([query])
    distances, indices = index.search(query_embedding, fetch_k)

    candidates = [(i, df_unique.iloc[i]['full_text']) for i in indices[0]]

    rerank_inputs = [(query, text) for _, text in candidates]
    scores = cross_encoder.predict(rerank_inputs)

    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
    return reranked

while True:
    query = input("\nIntroduce tu búsqueda (o escribe 'salir' para terminar): ")
    if query.lower() == 'salir':
        break

    results = semantic_search(query, k=10, fetch_k=30)

    print(f"\nResultados re-rankeados para la consulta: '{query}'")
    for (i, full_text), score in results:
        card_name = df_unique.iloc[i]['name']
        card_text = df_unique.iloc[i]['text'].replace('\n', ' ')
        print(f"- Score: {score:.2f} | {card_name}: '{card_text}'")
