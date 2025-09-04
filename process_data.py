import json
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import time

DATAFRAME_FILE = '/app/data/unique_cards.pkl'
INDEX_FILE = '/app/data/cards_faiss.index'

print("Cargando modelos...")
bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
# bi_encoder = SentenceTransformer('all-mpnet-base-v2') 
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
    df_unique['type'] = df_unique['type'].fillna('')
    df_unique['manaCost'] = df_unique['manaCost'].fillna('')
    indexed_texts = []
    card_metadata = []

    for idx, row in df_unique.iterrows():
        paragraphs = row['text'].split('\n')
        
        for p in paragraphs:
            p = p.strip()
            if p: 
                full_text = f"{p} {row['type']} {row['manaCost']}"
                indexed_texts.append(full_text)
                
                card_metadata.append(idx)

    df_indexed_metadata = pd.DataFrame(card_metadata, columns=['original_index'])
    
    print(f"Procesando {len(df_unique)} cartas únicas...")
    print(f"Indexando {len(indexed_texts)} párrafos y descripciones...")
    print("Generando embeddings (esto puede tardar varios minutos)...")
    
    card_embeddings = bi_encoder.encode(indexed_texts, show_progress_bar=True)

    print("Creando índice Faiss...")
    index = faiss.IndexFlatL2(card_embeddings.shape[1])
    index.add(card_embeddings)

    print("Guardando en disco...")
    if not os.path.exists(os.path.dirname(DATAFRAME_FILE)):
        os.makedirs(os.path.dirname(DATAFRAME_FILE))

    df_unique.to_pickle(DATAFRAME_FILE)
    df_indexed_metadata.to_pickle('/app/data/indexed_metadata.pkl')
    faiss.write_index(index, INDEX_FILE)
    print("¡Procesamiento inicial completado!")