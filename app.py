import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import requests

DATAFRAME_FILE = '/app/data/unique_cards.pkl'
INDEX_FILE = '/app/data/cards_faiss.index'

@st.cache_resource
def load_resources():
    print("Loading models and data...")
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # bi_encoder = SentenceTransformer('all-mpnet-base-v2') 
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    df_unique = pd.read_pickle(DATAFRAME_FILE)
    index = faiss.read_index(INDEX_FILE)
    return bi_encoder, cross_encoder, df_unique, index

@st.cache_data(ttl=3600) 
def get_card_image(card_name):
    try:
        response = requests.get(f"https://api.scryfall.com/cards/named?fuzzy={card_name}")
        response.raise_for_status()  
        data = response.json()
        return data['image_uris']['normal']
    except requests.exceptions.RequestException as e:
        st.warning(f"No se pudo obtener la imagen para {card_name}.")
        return None
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

st.title("MTG Search")

search_type_str = st.radio(
    "Selecciona el tipo de búsqueda:",
    ('Por Carta (c)', 'Por Descripción (d)')
)
search_type = 'c' if 'Carta' in search_type_str else 'd'

query = st.text_input(f"Introduce {'el nombre de la carta' if search_type == 'c' else 'la descripción'}: ")

if st.button("Buscar"):
    if query:
        with st.spinner('Buscando cartas...'):
            st.subheader(f"Resultados para la consulta: '{query}'")
            results = semantic_search(query, search_type, k=10, fetch_k=40)
                
            for (i, full_text), score in results:
                card_name = df_unique.iloc[i]['name']
                card_text = df_unique.iloc[i]['text'].replace('\n', ' ')
                card_mana_cost = df_unique.iloc[i]['manaCost']
                card_type = df_unique.iloc[i]['type']
                card_rarity = df_unique.iloc[i]['rarity']
                
                with st.expander(f"**{card_name}**"):
                    col1, col2 = st.columns([1, 2],gap="large")
                    
                    with col1:
                        image_url = get_card_image(card_name)
                        if image_url:
                            st.image(image_url, caption=card_name, width=250)
                        

                    with col2:
                        st.write(f"**Costo de Maná**: {card_mana_cost}")
                        st.write(f"**Tipo**: {card_type}")
                        st.write(f"**Rareza**: {card_rarity}")
                        st.write(f"**Texto**: {card_text}")
                        st.write(f"**Score de Relevancia**: {score:.2f}")

    else:
        st.error("Por favor, introduce un término de búsqueda.")