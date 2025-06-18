import os
import re
import gc
import torch
import random
import numpy as np
import pandas as pd
from collections import Counter
from fuzzywuzzy import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device configuration
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_path = os.path.join("data")
results_path = os.path.join("results", "metrics_s1.csv")
model_name = "microsoft/Phi-3.5-mini-instruct"

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    force_download=True
).to(device)
print("Model and tokenizer loaded successfully.")

# Load data
print("Loading datasets...")
df_users_ml = pd.read_csv(os.path.join(data_path, "df_user_ml-1m.csv"))
df_items_ml = pd.read_csv(os.path.join(data_path, "df_item_ml-1m.csv"))
test_data = pd.read_csv(os.path.join(data_path, "test_data_ml1m_fullInteraction_80users.csv"))
print("Datasets loaded.")

# Utility functions
def normalize_list(lst):
    return [re.sub(r'[^a-zA-Z0-9 ]', '', s.lower().strip()) for s in lst]

def get_recommendations(user_movies_string, candidate_movies):
    prompt = f"""
    Based on these movies: {user_movies_string}, recommend 10 movies that the user will likely enjoy.\n
    Choose only from the following list:\n{', '.join(candidate_movies)}\n
    Return only the movie titles, comma-separated, with no extra text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return [title.strip() for title in response.split(",") if title.strip() in candidate_movies]

# Evaluation Metrics
def hit_rate(recommendations, ground_truth, top_k=3, threshold=60):
    rec_norm = normalize_list(recommendations[:top_k])
    truth_norm = normalize_list(ground_truth)
    hits = sum(any(fuzz.partial_ratio(rec, truth) >= threshold for truth in truth_norm) for rec in rec_norm)
    return hits / top_k if top_k > 0 else 0

def average_rank(recommendations, ground_truth):
    rec_norm = normalize_list(recommendations)
    truth_norm = normalize_list(ground_truth)
    ranks = [next((idx + 1 for idx, rec in enumerate(rec_norm) if rec == item), None) for item in truth_norm]
    ranks = [r for r in ranks if r is not None]
    return sum(ranks) / len(ranks) if ranks else np.nan

def recall_at_k(preds, truths, k=5):
    return len(set(preds[:k]) & set(truths)) / len(set(truths)) if truths else 0.0

def dcg_at_k(recs, truths, k):
    return sum(1 / np.log2(i + 2) for i, item in enumerate(recs[:k]) if item in truths)

def ndcg_at_k(recs, truths, k):
    ideal_dcg = dcg_at_k(truths, truths, k)
    return dcg_at_k(recs, truths, k) / ideal_dcg if ideal_dcg != 0 else 0.0

def hhi(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    return sum((count / total) ** 2 for count in counter.values()) if total > 0 else 0

def entropy_score(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    return -sum((count / total) * np.log2(count / total) for count in counter.values() if count > 0) if total > 0 else 0

def gini_index(recommendations):
    if not recommendations:
        return 0
    values = np.array(list(Counter(recommendations).values()))
    values = values / values.sum()
    n = len(values)
    values.sort()
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))

# Evaluation loop
metrics = []
user_ids = test_data['userId'].unique()[:10]

for uid in user_ids:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    user_history = test_data[test_data['userId'] == uid]
    user_movie_titles = df_items_ml[df_items_ml['itemId'].isin(user_history['itemId'])]['title'].tolist()

    if len(user_movie_titles) < 6:
        continue

    input_movies = user_movie_titles[:3]
    ground_truth = user_movie_titles[3:]
    candidate_pool = df_items_ml[~df_items_ml['title'].isin(user_movie_titles)]['title']
    candidate_titles = candidate_pool.sample(n=120, random_state=42).tolist()
    full_movie_list = input_movies + candidate_titles

    user_movies_string = ', '.join(input_movies)
    recommendations = get_recommendations(user_movies_string, full_movie_list)

    metrics.append({
        "user_id": uid,
        "hit_rate": hit_rate(recommendations, ground_truth),
        "avg_rank": average_rank(recommendations, ground_truth),
        "recall@5": recall_at_k(recommendations, ground_truth, k=5),
        "ndcg@5": ndcg_at_k(recommendations, ground_truth, k=5),
        "hhi": hhi(recommendations),
        "entropy": entropy_score(recommendations),
        "gini": gini_index(recommendations)
    })

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(results_path, index=False)

print("Average Metrics for First 10 Users (S1):")
print(df_metrics.mean())
