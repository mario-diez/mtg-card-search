# streamlit_app.py
import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

DATAFRAME_FILE = 'unique_cards.pkl'
INDEX_FILE = 'cards_faiss.index'

@st.cache_resource
def load_resources():
    print("Loading models and data...")
    bi_encoder = SentenceTransformer('all-mpnet-base-v2') 
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    df_unique = pd.read_pickle(DATAFRAME_FILE)
    index = faiss.read_index(INDEX_FILE)
    return bi_encoder, cross_encoder, df_unique, index

bi_encoder, cross_encoder, df_unique, index = load_resources()

def semantic_search(query_text, search_type, k=10, fetch_k=30):
    if search_type == 'c':
        card_match = df_unique[df_unique['name'].str.lower() == query_text.lower()]
        if not card_match.empty:
            target_card_text = card_match.iloc[0]['full_text']
            st.info(f"Buscando cartas similares a '{card_match.iloc[0]['name']}'...")
            query_embedding = bi_encoder.encode([target_card_text])
            original_card_index = card_match.index[0]
            exclude_original = True
            rerank_query = target_card_text 
        else:
            st.warning("Carta no encontrada. Buscando por descripción en su lugar.")
            query_embedding = bi_encoder.encode([query_text])
            exclude_original = False
            rerank_query = query_text
    else:
        query_embedding = bi_encoder.encode([query_text])
        exclude_original = False
        rerank_query = query_text
        
    distances, indices = index.search(query_embedding, fetch_k)
    
    candidates = []
    for i in indices[0]:
        if exclude_original and i == original_card_index:
            continue
        candidates.append((i, df_unique.iloc[i]['full_text']))
        
    rerank_inputs = [(rerank_query, text) for _, text in candidates]
    scores = cross_encoder.predict(rerank_inputs)
    
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
    return reranked

st.title("Buscador de Cartas de Magic: The Gathering")

search_type_str = st.radio(
    "Selecciona el tipo de búsqueda:",
    ('Por Carta (c)', 'Por Descripción (d)')
)
search_type = 'c' if 'Carta' in search_type_str else 'd'

query = st.text_input(f"Introduce {'el nombre de la carta' if search_type == 'c' else 'la descripción'}: ")

if st.button("Buscar"):
    if query:
        st.subheader(f"Resultados para la consulta: '{query}'")
        results = semantic_search(query, search_type, k=10, fetch_k=30)
        
        for (i, full_text), score in results:
            card_name = df_unique.iloc[i]['name']
            card_text = df_unique.iloc[i]['text'].replace('\n', ' ')
            card_mana_cost = df_unique.iloc[i]['manaCost']
            card_type = df_unique.iloc[i]['type']
            card_rarity = df_unique.iloc[i]['rarity']
            
            st.write(f"- **Score**: {score:.2f} | **Mana Cost**: {card_mana_cost} | **Type**: {card_type} | **Text**: {card_text}")
            

    else:
        st.error("Por favor, introduce un término de búsqueda.")