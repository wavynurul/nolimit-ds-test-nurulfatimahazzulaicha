import streamlit as st
import pandas as pd
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import os

# Page configuration
st.set_page_config(
    page_title="Ticket Priority Predictor",
    page_icon="📩",
    layout="wide"
)

st.title("📩 Customer Support Ticket Prioritization")
st.markdown(
    """
Predict the priority of support tickets (High / Medium / Low) using DistilBERT embeddings and Logistic Regression.

You can either type a ticket manually or upload a CSV file with `Ticket Subject` and `Ticket Description` columns.
"""
)

# Base path for models (inside 'src' folder)
BASE_PATH = os.path.join(os.path.dirname(__file__), "src")

# Load models (cached for speed)
@st.cache_resource
def load_models():
    clf = joblib.load(os.path.join(BASE_PATH, "ticket_priority_clf.joblib"))
    le = joblib.load(os.path.join(BASE_PATH, "label_encoder.joblib"))
    tokenizer = joblib.load(os.path.join(BASE_PATH, "distilbert_tokenizer.joblib"))
    model = joblib.load(os.path.join(BASE_PATH, "distilbert_model.joblib"))
    return clf, le, tokenizer, model

clf, le, tokenizer, model = load_models()

# Function to embed texts
def embed_texts(texts, tokenizer, model, max_length=128):
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings)
    return torch.cat(embeddings).numpy()

# Input method
input_type = st.radio("Choose input method:", ("Type manually", "Upload CSV"))

# Manual input
if input_type == "Type manually":
    ticket_subject = st.text_input("Enter ticket subject:")
    ticket_description = st.text_area("Enter ticket description:")
    if st.button("Predict Priority"):
        if ticket_subject.strip() or ticket_description.strip():
            with st.spinner("Predicting..."):
                combined_text = f"{ticket_subject} {ticket_description}"
                emb = embed_texts([combined_text], tokenizer, model)
                pred_label = clf.predict(emb)
                pred_priority = le.inverse_transform(pred_label)[0]
            st.success(f"✅ Predicted Priority: **{pred_priority}**")
        else:
            st.warning("⚠️ Please enter subject or description!")

# CSV upload
elif input_type == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload CSV file with 'Ticket Subject' and 'Ticket Description' columns:", type=["csv"]
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Ticket Subject' not in df.columns or 'Ticket Description' not in df.columns:
            st.error("❌ CSV must have 'Ticket Subject' and 'Ticket Description' columns")
        else:
            # Combine subject + description
            df['TicketText'] = df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna('')

            # Use a form to keep the file
            with st.form("predict_form"):
                submit_button = st.form_submit_button("Predict All Tickets")
                if submit_button:
                    with st.spinner("Predicting all tickets..."):
                        emb = embed_texts(df['TicketText'].tolist(), tokenizer, model)
                        pred_labels = clf.predict(emb)
                        df['PredictedPriority'] = le.inverse_transform(pred_labels)
                    st.success("✅ Prediction completed!")
                    st.dataframe(df[['Ticket Subject', 'Ticket Description', 'PredictedPriority']])

                    # Download CSV
                    csv = df[['Ticket Subject', 'Ticket Description', 'PredictedPriority']].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime='text/csv'
                    )
