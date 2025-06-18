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
results_path = os.path.join("results", "s6_metrics.csv")
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

def entropy(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * np.log2(count / total) for count in counter.values() if count > 0)

def gini_index(x):
    x = np.array(x, dtype=np.float64)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-6
    x_sorted = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def recall(predictions, ground_truth, k=10):
    if not ground_truth:
        return 0.0
    hits = len(set(predictions[:k]) & set(ground_truth))
    return hits / len(ground_truth)

def dcg_at_k(recs, truths, k):
    dcg = 0.0
    for i, item in enumerate(recs[:k]):
        if item in truths:
            dcg += 1 / np.log2(i + 2)
    return dcg

def ndcg(predictions, ground_truth, k=10):
    ideal_dcg = dcg_at_k(ground_truth, ground_truth, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(predictions, ground_truth, k) / ideal_dcg

# S6 Motivate Reasoning recommendation function
def get_recommendations_s6_motivate(user_history, df_items, candidate_movies_df, k=10):
    merged = pd.merge(user_history, df_items, on='itemId')

    user_mean_rating = user_history['rating'].mean()
    genre_scores = {}
    director_scores = {}
    genre_count = {}

    # Build user movie string for prompt
    watched_titles = df_items[df_items['itemId'].isin(user_history['itemId'])]['title'].tolist()
    user_movies_string = ", ".join(watched_titles)

    for _, row in merged.iterrows():
        centered_rating = row['rating'] - user_mean_rating
        for genre in row['genres'].split('|'):
            genre_scores[genre] = genre_scores.get(genre, 0) + centered_rating
            genre_count[genre] = genre_count.get(genre, 0) + 1
        director = row.get('director', None)
        if director:
            director_scores[director] = director_scores.get(director, 0) + centered_rating

    results = []
    for _, row in candidate_movies_df.iterrows():
        movie_genres = row['genres'].split('|')
        director = row.get('director', 'Unknown')

        genre_match_score = sum(genre_scores.get(g, 0) for g in movie_genres)
        director_score = director_scores.get(director, 0)
        genre_match_count = sum(1 for g in movie_genres if g in genre_scores)

        # Exploration bonus: favor rare genres user watched less often
        exploration_bonus = sum(1 / (1 + genre_count.get(g, 1)) for g in movie_genres)

        # Total motivation-focused score
        score = 0.6 * genre_match_score + 0.2 * director_score + 0.1 * genre_match_count + 0.1 * exploration_bonus

        # Rationale generation
        genre_matches = [f"{g} (score: {genre_scores[g]:.1f})" for g in movie_genres if g in genre_scores]
        director_match = f"frequent director: {director} (score: {director_score:.1f})" if director_score > 0 else ""
        rationale_parts = genre_matches + ([director_match] if director_match else [])

        if rationale_parts:
            if len(genre_matches) >= 2:
                rationale = f"'{row['title']}' aligns well with your favorite genres: {', '.join(genre_matches)}."
            else:
                rationale = f"Based on your preferences in {', '.join(rationale_parts)}, we recommend '{row['title']}'."
        else:
            rationale = f"'{row['title']}' is suggested to broaden your movie taste."

        results.append((row['title'], score, rationale))

    topk = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    titles, scores, rationales = zip(*topk)

    # Construct prompt (optional, useful for explainability or future LLM integration)
    prompt = f"Provide {k} carefully selected movie recommendations, each accompanied by a rationale explaining its suitability for the user’s preferences. The user has previously enjoyed the following movies: {user_movies_string}"

    return list(titles), list(scores), list(rationales), prompt

# Run evaluation on a subset of users
user_ids_s6 = test_data_ml1m_fullInteraction['userId'].unique()[:10]
metrics_s6 = []
all_recommendations_s6 = []

for uid in user_ids_s6:
    hist = test_data_ml1m_fullInteraction[test_data_ml1m_fullInteraction['userId'] == uid]
    titles_all = df_items_ml[df_items_ml['itemId'].isin(hist['itemId'])]['title'].tolist()
    if len(titles_all) < 6:
        continue

    input_titles = titles_all[:5]
    truth = titles_all[5:]

    # Candidate pool: exclude already watched titles
    candidate_movies_df_s6 = df_items_ml[~df_items_ml['title'].isin(input_titles)].copy()

    # Popularity normalization (optional)
    pop_series = test_data_ml1m_fullInteraction['itemId'].value_counts()
    candidate_movies_df_s6['popularity'] = candidate_movies_df_s6['itemId'].map(pop_series).fillna(0)
    max_pop = candidate_movies_df_s6['popularity'].max()
    candidate_movies_df_s6['normalized_popularity'] = candidate_movies_df_s6['popularity'] / max_pop if max_pop > 0 else 0

    # Downsample candidates for speed
    candidate_movies_df_s6 = candidate_movies_df_s6.sample(min(len(candidate_movies_df_s6), 120), random_state=42)

    recs_titles, recs_scores, recs_rationales, prompt = get_recommendations_s6_motivate(
        hist, df_items_ml, candidate_movies_df_s6
    )

    all_recommendations_s6.extend(recs_titles)

    # Calculate metrics
    metrics_s6.append({
        "hit_rate": hit_rate(recs_titles, truth),
        "avg_rank": average_rank(recs_titles, truth),
        "hhi": hhi(recs_titles),
        "recall@5": recall(recs_titles, truth, k=5),
        "ndcg@5": ndcg(recs_titles, truth, k=5),
        "entropy": entropy(recs_titles),
        "gini": gini_index(recs_scores)
    })

# Summary report
df_metrics_s6 = pd.DataFrame(metrics_s6)
print("Average Metrics for First 10 Users (S6 Motivate Reasoning):")
print(df_metrics_s6.mean())

print("\nSystem-Level Entropy Across All S6 Recommendations:")
print(entropy(all_recommendations_s6))

# Save metrics to CSV
os.makedirs("results", exist_ok=True)
df_metrics_s6.to_csv(results_path, index=True)
print(f"\nSaved metrics to {results_path}")
