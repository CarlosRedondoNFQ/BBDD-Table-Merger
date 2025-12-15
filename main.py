import os
import logging
import streamlit as st
import pandas as pd
import numpy as np
# from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the SentenceTransformer model once and cache it
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error("Failed to load SentenceTransformer model", exc_info=True)
        st.error("Model loading error.")
        raise e

MODEL = load_sentence_transformer()

def cargar_excel(ruta):
    try:
        data = pd.read_excel(ruta)
        df = pd.DataFrame(data)
        # Validate required columns
        required_cols = ['bt_desc_eng', 'bt_desc_esp', 'bt_name', 'origin', 'ID']
        if not all(col in df.columns for col in required_cols):
            missing = set(required_cols) - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        return df
    except Exception as e:
        logger.error("Error loading Excel file", exc_info=True)
        st.error("Error processing the uploaded file.")
        raise e

def preprocess_text(df):
    for col in ['bt_desc_eng', 'bt_desc_esp', 'bt_name']:
        df[col] = df[col].str.lower()
    return df

def generate_embeddings(df):
    # Use the globally loaded MODEL
    def obtener_embedding(texto):
        try:
            return MODEL.encode(texto)
        except Exception as e:
            logger.error("Error generating embedding", exc_info=True)
            return np.zeros(384)  # or handle appropriately

    # Check and ensure the correct mapping between language and column
    # (Assuming bt_desc_eng should be English and bt_desc_esp Spanish)
    df['emb_eng'] = df['bt_desc_eng'].apply(obtener_embedding)
    df['emb_esp'] = df['bt_desc_esp'].apply(obtener_embedding)
    return df

def prepare_data(df):
    processed_df = preprocess_text(df)
    embeddings_df = generate_embeddings(processed_df)
    return embeddings_df

def find_matches(path):
    data = cargar_excel(path)
    df = prepare_data(data)
    all_matches = []
    
    # Iterate over all unique pairs; consider optimizations for large datasets
    for idx1, row1 in df.iterrows():
        for idx2, row2 in df.iterrows():
            if row1['origin'] == row2['origin']:
                continue  # Skip same database

            name_fuzz_similarity = fuzz.partial_ratio(row1['bt_name'], row2['bt_name'])
            desc_similarity_eng = cosine_similarity([row1['emb_eng']], [row2['emb_eng']])[0][0]
            desc_similarity_esp = cosine_similarity([row1['emb_esp']], [row2['emb_esp']])[0][0]
            
            # Combine similarity scores with configurable weights
            combined_score = (0.4 * name_fuzz_similarity +
                              0.6 * (max(desc_similarity_eng, desc_similarity_esp) * 100))
            
            all_matches.append({
                'table1': row1['bt_name'], 'db1': row1['origin'], 'id1': row1['ID'],
                'table2': row2['bt_name'], 'db2': row2['origin'], 'id2': row2['ID'],
                'Custom score': combined_score,
                'Description similarity (English)': desc_similarity_eng,
                'Description similarity (Spanish)': desc_similarity_esp,
                'Name similarity': name_fuzz_similarity
            })
    
    matches_df = pd.DataFrame(all_matches)
    sorted_matches = matches_df.sort_values('Custom score', ascending=False)
    
    # Remove duplicate matches based on IDs
    seen_ids = set()
    clean_matches = []
    for _, row in sorted_matches.iterrows():
        if row['id1'] in seen_ids or row['id2'] in seen_ids:
            continue
        seen_ids.update([row['id1'], row['id2']])
        clean_matches.append({
            'ID1': row['id1'], 'Table1': row['table1'],
            'ID2': row['id2'], 'Table2': row['table2'],
            'Custom Score': row['Custom score'],
            'Description similarity (English)': row['Description similarity (English)'],
            'Description similarity (Spanish)': row['Description similarity (Spanish)'],
            'Name similarity': row['Name similarity']
        })
    
    flag = len(seen_ids) == df['ID'].nunique()
    logger.info(f"All tables were matched: {flag}")
    st.write(f"All tables were matched: {flag}")
    return pd.DataFrame(clean_matches)

# Streamlit UI setup
st.title("DataBase Matcher")
uploaded_file = st.file_uploader("Upload your file with the database table names", type="xlsx")

if uploaded_file:
    try:
        # Ensure directory exists
        upload_dir = "uploaded_files"
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, uploaded_file.name)
        with open(path, "wb") as file:
            file.write(uploaded_file.read())

        matches_df = find_matches(path)
        # output_path = "final_table_matches.xlsx"
        # matches_df.to_excel(output_path, index=False)
        # st.write(f"Saved in {output_path}")
        st.dataframe(matches_df)
    except Exception as e:
        logger.error("Error during processing", exc_info=True)
        st.error("An error occurred during processing. Please check the logs.")
