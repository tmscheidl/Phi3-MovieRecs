import os
import re
import gc
import torch
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz

# Device configuration (not really needed for S3, but keeping style)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_path = os.path.join("data")
results_path = os.path.join("results", "s3_metrics.csv")

# Load data
print("Loading datasets...")
df_users_ml = pd.read_csv(os.path.join(data_path, "df_user_ml-1m.csv"))
df_items_ml = pd.read_csv(os.path.join(data_path, "df_item_ml-1m.csv"))
test_data_ml1m_fullInteraction = pd.read_csv(os.path.join(data_path, "test_data_ml1m_fullInteraction_80users.csv"))
print("Datasets loaded.")

# --- Helper Functions ---
def normalize_list(lst):
    return [re.sub(r'[^a-zA-Z0-9 ]', '', s.lower().strip()) for s in lst]

def hit_rate(recommendations, ground_truth, top_k=3, threshold=60):
    recommendations_norm = normalize_list(recommendations)
    ground_truth_norm = normalize_list(ground_truth)
    hits = sum(
        any(fuzz.partial_ratio(rec, truth) >= threshold for truth in ground_truth_norm)
        for rec in recommendations_norm[:top_k]
    )
    return hits / top_k if top_k > 0 else 0

def average_rank(recommendations, ground_truth):
    recommendations_norm = normalize_list(recommendations)
    ground_truth_norm = normalize_list(ground_truth)
    ranks = [
        next((idx + 1 for idx, rec in enumerate(recommendations_norm) if rec == item), None)
        for item in ground_truth_norm
    ]
    ranks = [r for r in ranks if r is not None]
    if len(ranks) == 0:
        return np.nan
    return sum(ranks) / len(ranks)

def hhi(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    return sum((count / total) ** 2 for count in counter.values())

def entropy_metric(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * np.log2(count / total) for count in counter.values() if count > 0)

def gini_index_scores(x):
    x = np.array(x)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-6
    x_sorted = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def recall_at_k(preds, truths, k=3):
    preds_at_k = preds[:k]
    hits = len(set(preds_at_k) & set(truths))
    return hits / len(set(truths)) if truths else 0.0

def dcg_at_k(recs, truths, k):
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        if item in truths:
            dcg += 1 / np.log2(i + 2)
    return dcg

def ndcg_at_k(recs, truths, k):
    ideal_dcg = dcg_at_k(truths, truths, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(recs, truths, k) / ideal_dcg

# --- S3 Diversify Recommendation Function ---
def get_recommendations_s3_diversify(user_history, df_items, candidate_movies_df, k=10):
    # Compute genre scores from user history
    merged = pd.merge(user_history, df_items, on='itemId')
    genre_scores = {}
    for _, r in merged.iterrows():
        for g in r['genres'].split('|'):
            genre_scores[g] = genre_scores.get(g, 0) + r['rating']

    # Score candidate movies
    results = []
    for idx, r in candidate_movies_df.iterrows():
        gs = sum(genre_scores.get(g, 0) for g in r['genres'].split('|'))
        pop = r['normalized_popularity']
        noise = np.random.normal(0, 0.5)
        bias = idx * 1e-4
        score = gs + 0.7 * pop + noise + bias
        results.append((r['title'], score))

    # Select top-k
    topk = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    titles, raw_scores = zip(*topk)

    # Normalize for Gini
    raw_scores = np.array(raw_scores)
    if np.std(raw_scores) < 1e-3:
        raw_scores += np.random.normal(0, 0.1, size=len(raw_scores))
    raw_scores -= raw_scores.min()
    raw_scores += 1e-6
    scores = raw_scores.tolist()

    return list(titles), scores

# --- Evaluation Loop ---
user_ids_s3 = test_data_ml1m_fullInteraction['userId'].unique()[:10]
metrics_s3 = []

for uid in user_ids_s3:
    hist = test_data_ml1m_fullInteraction[test_data_ml1m_fullInteraction['userId'] == uid]
    titles_all = df_items_ml[df_items_ml['itemId'].isin(hist['itemId'])]['title'].tolist()

    if len(titles_all) < 6:
        continue

    input_titles = titles_all[:5]
    truth = titles_all[5:]

    # Candidate movies filtered by top genres
    user_genres = df_items_ml[df_items_ml['title'].isin(titles_all)]['genres'].str.split('|').explode().value_counts()
    top_genres = user_genres.index[:3]
    mask = df_items_ml['genres'].apply(lambda g: any(genre in g.split('|') for genre in top_genres))
    candidate_movies_df_s3 = df_items_ml[mask & (~df_items_ml['title'].isin(input_titles))].copy()

    # Normalize popularity
    movie_popularity = test_data_ml1m_fullInteraction['itemId'].value_counts()
    candidate_movies_df_s3['popularity'] = candidate_movies_df_s3['itemId'].map(movie_popularity).fillna(0)
    max_popularity = candidate_movies_df_s3['popularity'].max()
    candidate_movies_df_s3['normalized_popularity'] = candidate_movies_df_s3['popularity'] / max_popularity
    candidate_movies_df_s3['normalized_popularity'] += np.random.normal(0, 0.01, size=len(candidate_movies_df_s3))

    # Downsample to limit size
    candidate_movies_df_s3 = candidate_movies_df_s3.sample(min(len(candidate_movies_df_s3), 120), random_state=42)

    # Get recommendations
    recs_titles, recs_scores = get_recommendations_s3_diversify(hist, df_items_ml, candidate_movies_df_s3)

    metrics_s3.append({
        "hit_rate": hit_rate(recs_titles, truth),
        "avg_rank": average_rank(recs_titles, truth),
        "recall@5": recall_at_k(recs_titles, truth, k=5),
        "ndcg@5": ndcg_at_k(recs_titles, truth, k=5),
        "hhi": hhi(recs_titles),
        "entropy": entropy_metric(recs_titles),
        "gini": gini_index_scores(recs_scores)
    })

# --- Metrics Summary ---
df_metrics_s3 = pd.DataFrame(metrics_s3)
print("Average Metrics for First 10 Users (S3 Diversify):")
print(df_metrics_s3.mean())

# Save metrics
os.makedirs("results", exist_ok=True)
df_metrics_s3.to_csv(results_path, index=True)
print("Saved metrics to", results_path)
