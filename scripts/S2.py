import os
import re
import gc
import torch
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import ndcg_score
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
test_data_ml1m_fullInteraction = pd.read_csv(os.path.join(data_path, "test_data_ml1m_fullInteraction_80users.csv"))
print("Datasets loaded.")

# --- Gini Index ---
def gini_index_scores(x):
    x = np.array(x, dtype=np.float64)  # Force float dtype at the start
    min_val = np.amin(x)
    if min_val < 0:
        x = x - min_val  # avoid in-place subtraction
    x = x + 1e-6  # avoid zero division, no in-place addition
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    
# --- Metrics ---
def hit_rate(recommended, ground_truth):
    return any(item in ground_truth for item in recommended)

def average_rank(recommended, ground_truth):
    ranks = [recommended.index(item) + 1 for item in ground_truth if item in recommended]
    return np.mean(ranks) if ranks else len(recommended) + 1

def hhi(recommended):
    counts = Counter(recommended)
    total = sum(counts.values())
    return sum((count / total) ** 2 for count in counts.values())

def entropy(recommended):
    counts = Counter(recommended)
    total = sum(counts.values())
    return -sum((count / total) * np.log2(count / total) for count in counts.values() if count > 0)

def recall_at_k(recommended, ground_truth, k):
    recommended_top_k = recommended[:k]
    hits = sum(1 for item in ground_truth if item in recommended_top_k)
    return hits / len(ground_truth) if ground_truth else 0.0

def ndcg_at_k(recommended, ground_truth, k):
    relevance = [1 if item in ground_truth else 0 for item in recommended[:k]]
    return ndcg_score([relevance], [sorted(relevance, reverse=True)])

# --- Recommendation Model ---
def get_recommendations_genre_strict(user_history, df_items, candidate_movies_df):
    # Get top genres from user history
    user_genres = df_items[df_items['itemId'].isin(user_history['itemId'])]['genres'].str.split('|').explode().value_counts()
    top_genres = user_genres.index[:3]

    results = []
    for _, row in candidate_movies_df.iterrows():
        genres = set(row['genres'].split('|'))
        genre_match = len(genres & set(top_genres))
        popularity = row['normalized_popularity']
        score = genre_match + 0.1 * popularity  # Tunable weight
        results.append((row['title'], score))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    recommended_titles = [title for title, _ in sorted_results[:10]]

    # Prompt (if needed)
    user_movies_string = ", ".join(df_items[df_items['itemId'].isin(user_history['itemId'])]['title'].tolist())
    prompt = (
        f"Provide 10 movie recommendations that strictly match the user's favorite genres.\n"
        f"User history includes: {user_movies_string}.\n"
        f"Focus on genre overlap with top genres: {', '.join(top_genres)}."
    )

    return recommended_titles, prompt

# --- Evaluation Loop ---
user_ids_s2 = test_data_ml1m_fullInteraction['userId'].unique()[:10]
metrics_s2 = []
all_rec_titles = []
all_ground_truths = []
all_user_recs = []

try:
    for user_id in user_ids_s2:
        user_history = test_data_ml1m_fullInteraction[test_data_ml1m_fullInteraction['userId'] == user_id]
        user_movie_titles = df_items_ml[df_items_ml['itemId'].isin(user_history['itemId'])]['title'].tolist()
        
        if len(user_movie_titles) < 6:
            continue
        
        input_movies = user_movie_titles[:5]
        ground_truth = user_movie_titles[5:]
        all_ground_truths.append(ground_truth)

        user_genres = df_items_ml[df_items_ml['title'].isin(user_movie_titles)]['genres'].str.split('|').explode().value_counts()
        top_genres = user_genres.index[:3]

        mask = df_items_ml['genres'].apply(lambda g: any(genre in g.split('|') for genre in top_genres))
        candidate_movies_df = df_items_ml[mask & (~df_items_ml['title'].isin(input_movies))].copy()

        movie_popularity = test_data_ml1m_fullInteraction['itemId'].value_counts()
        candidate_movies_df['popularity'] = candidate_movies_df['itemId'].map(movie_popularity).fillna(0)
        max_popularity = candidate_movies_df['popularity'].max()
        candidate_movies_df['normalized_popularity'] = candidate_movies_df['popularity'] / max_popularity
        candidate_movies_df['normalized_popularity'] += np.random.normal(0, 0.01, size=len(candidate_movies_df))

        candidate_movies_df = candidate_movies_df.sample(min(len(candidate_movies_df), 120), random_state=42)

        recs = get_recommendations_genre_strict(user_history, df_items_ml, candidate_movies_df)
        rec_titles = recs[0] if isinstance(recs, tuple) else recs
        all_rec_titles.extend(rec_titles)
        all_user_recs.append(rec_titles)

        metrics_s2.append({
            "hit_rate": hit_rate(rec_titles, ground_truth),
            "avg_rank": average_rank(rec_titles, ground_truth),
            "hhi": hhi(rec_titles),
            "entropy": entropy(rec_titles)
        })
except Exception as e:
    print(f"Error during evaluation: {e}")

# --- Global Gini Index ---
rec_title_counts = Counter(all_rec_titles)
gini_val = gini_index_scores(list(rec_title_counts.values()))

# --- Metrics DataFrame ---
df_metrics = pd.DataFrame(metrics_s2)
df_metrics.loc["Average"] = df_metrics.mean()
df_metrics.at["Average", "gini"] = gini_val

# --- Global Recall@5 and NDCG@5 ---
recall_scores, ndcg_scores = [], []
for rec_titles, ground_truth in zip(all_user_recs, all_ground_truths):
    rec_norm = [title.lower() for title in rec_titles]
    ground_truth_norm = [title.lower() for title in ground_truth]
    
    recall_scores.append(recall_at_k(rec_norm, ground_truth_norm, k=5))
    ndcg_scores.append(ndcg_at_k(rec_norm, ground_truth_norm, k=5))

df_metrics.at["Average", "recall@5"] = np.mean(recall_scores)
df_metrics.at["Average", "ndcg@5"] = np.mean(ndcg_scores)

# --- Final Output ---
print("\nAverage Metrics for First 10 Users:")
print(df_metrics.loc["Average"])

# Save metrics to a CSV file in the results folder
os.makedirs("results", exist_ok=True)
df_metrics.to_csv("results/s2_metrics.csv", index=True)
print("\nSaved metrics to results/s2_metrics.csv")
