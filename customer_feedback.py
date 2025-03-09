import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import hdbscan
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Local Model (Mistral)
model_name = "mistralai/mistralai/Mistral-7B-v0.1"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

# Text Cleaning Function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    return " ".join([word for word in words if word.isalnum() and word not in stop_words])

# Extract Themes using NMF
def extract_themes(feedbacks, num_topics=5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(feedbacks)
    nmf = NMF(n_components=num_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    feature_names = vectorizer.get_feature_names_out()
    themes = [" ".join([feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]) for topic in H]
    return themes

# HDBSCAN Clustering
def cluster_feedback(feedbacks):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(feedbacks)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric='euclidean')
    clusters = clusterer.fit_predict(X.toarray())
    return clusters

# Generate Cluster Name using LLM
def generate_cluster_name(cluster_text):
    prompt = f"Generate a short theme name for this feedback cluster: {cluster_text}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Process Feedback from Excel
def process_feedback(file_path):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    df['Cleaned_Feedback'] = df['Feedback'].apply(clean_text)
    df['Overall_Theme'] = extract_themes(df['Cleaned_Feedback'])
    df['Cluster'] = cluster_feedback(df['Cleaned_Feedback'])

    theme_names = {}
    for cluster in set(df['Cluster']):
        if cluster == -1:
            continue  # Skip noise points
        cluster_text = " ".join(df[df['Cluster'] == cluster]['Cleaned_Feedback'])
        theme_names[cluster] = generate_cluster_name(cluster_text)

    df['Clustered_Theme'] = df['Cluster'].map(theme_names)

    output_file = "clustered_feedback.xlsx"
    df.to_excel(output_file, index=False)
    print(f"✅ Clustering and theme naming complete! Saved to '{output_file}'.")

# Run the script
process_feedback("customer_feedback.xlsx")
