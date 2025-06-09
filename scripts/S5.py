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
results_path = os.path.join("results", "s5_metrics.csv")
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

# Helper functions
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

# Prompt function for S5 Surprise
def get_prompt_s5_surprise(input_titles, df_items):
    top_genres = df_items[df_items['title'].isin(input_titles)]['genres'].str.split('|').explode().value_counts().index[:3]
    genres_str = ', '.join(top_genres)
    prompt = f"""
You are a movie recommender. Your task is to recommend films that surprise the user by avoiding mainstream blockbusters.
The user has watched and liked the following movies: {', '.join(input_titles)}.
These seem to fall into the genres: {genres_str}.

Your goal is to suggest 10 lesser-known, high-quality films that are likely to delight the user, but aren't predictable.
Avoid overly popular titles. Favor unique, unconventional, or underrated works.
Do not repeat any of the user's already-watched movies.
For each recommendation, include the title only.
"""
    return prompt.strip()

# Recommender function for S5 Surprise
def get_recommendations_s5_surprise(user_history, df_items, candidate_movies_df, k=10):
    # 1) Genre weights (downweighted for surprise)
    merged = pd.merge(user_history, df_items, on='itemId')
    genre_scores = {}
    for _, r in merged.iterrows():
        for g in r['genres'].split('|'):
            genre_scores[g] = genre_scores.get(g, 0) + r['rating']

    # 2) Score candidates
    results = []
    for idx, r in candidate_movies_df.iterrows():
        genres = r['genres'].split('|')
        gs = sum(genre_scores.get(g, 0) for g in genres)

        # Heavily penalize popularity (promote obscure films)
        pop_penalty = -2.0 * np.log1p(r['normalized_popularity'] + 1e-6)

        # Strong noise to introduce surprise
        noise = np.random.normal(0, 0.6)

        # Final score: de-emphasize genre score, maximize surprise
        score = 0.5 * gs + pop_penalty + noise + idx * 1e-4
        results.append((r['title'], score))

    # 3) Take top-k
    topk = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    titles, scores = zip(*topk)
    return list(titles), list(scores)

# Run recommendation and evaluation for first 10 users
user_ids_s5 = test_data_ml1m_fullInteraction['userId'].unique()[:10]
metrics_s5 = []
all_recommendations_s5 = []  # To collect all recommended titles for global entropy

for uid in user_ids_s5:
    hist = test_data_ml1m_fullInteraction[test_data_ml1m_fullInteraction['userId'] == uid]
    titles_all = df_items_ml[df_items_ml['itemId'].isin(hist['itemId'])]['title'].tolist()
    if len(titles_all) < 6:
        continue

    input_titles = titles_all[:5]
    truth = titles_all[5:]

    # Sample from broader pool — no strict genre filtering for surprise
    candidate_movies_df_s5 = df_items_ml[~df_items_ml['title'].isin(input_titles)].copy()

    # Normalize and add noise to popularity
    pop_series = test_data_ml1m_fullInteraction['itemId'].value_counts()
    candidate_movies_df_s5['popularity'] = candidate_movies_df_s5['itemId'].map(pop_series).fillna(0)
    max_pop = candidate_movies_df_s5['popularity'].max()
    candidate_movies_df_s5['normalized_popularity'] = candidate_movies_df_s5['popularity'] / max_pop
    candidate_movies_df_s5['normalized_popularity'] += np.random.normal(0, 0.01, size=len(candidate_movies_df_s5))

    candidate_movies_df_s5 = candidate_movies_df_s5.sample(min(len(candidate_movies_df_s5), 120), random_state=42)

    recs_titles, recs_scores = get_recommendations_s5_surprise(
        hist, df_items_ml, candidate_movies_df_s5
    )

    # Append to global recommendation list
    all_recommendations_s5.extend(recs_titles)

    # Per-user metrics
    metrics_s5.append({
        "hit_rate": hit_rate(recs_titles, truth),
        "avg_rank": average_rank(recs_titles, truth),
        "recall@5": recall_at_k(recs_titles, truth, k=5),
        "ndcg@5": ndcg_at_k(recs_titles, truth, k=5),
        "hhi": hhi(recs_titles),
        "entropy": entropy_metric(recs_titles),
        "gini": gini_index_scores(recs_scores)
    })

# Convert and print average per-user metrics
df_metrics_s5 = pd.DataFrame(metrics_s5)
print("Average Metrics for First 10 Users (S5 Surprise):")
print(df_metrics_s5.mean())

# Compute and print system-level entropy
print("\nSystem-Level Entropy Across All S5 Recommendations:")
print(entropy_metric(all_recommendations_s5))

# Save metrics to a CSV file in the results folder
os.makedirs("results", exist_ok=True)
df_metrics_s5.to_csv(results_path, index=True)
print("\nSaved metrics to", results_path)