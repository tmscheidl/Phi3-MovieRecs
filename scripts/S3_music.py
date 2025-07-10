import os
import numpy as np
import pandas as pd
import torch
from collections import Counter

# Paths
data_path = os.path.join("data")
results_path = os.path.join("results", "s3_music_metrics.csv")
model_name = "microsoft/Phi-3.5-mini-instruct"

# --- Evaluation Metrics ---
def hit_rate(recommendations, ground_truth, top_k=3, threshold=60):
    recs = [r.lower().strip() for r in recommendations[:top_k]]
    truths = [t.lower().strip() for t in ground_truth]
    hits = sum(any(rec in t or t in rec for t in truths) for rec in recs)
    return hits / top_k if top_k > 0 else 0.0

def average_rank(recommendations, ground_truth):
    recs = [r.lower().strip() for r in recommendations]
    truths = [t.lower().strip() for t in ground_truth]
    ranks = [i+1 for i, rec in enumerate(recs) if rec in truths]
    return np.mean(ranks) if ranks else np.nan

def recall_at_k(preds, truths, k=3):
    return len(set(preds[:k]) & set(truths)) / len(set(truths)) if truths else 0.0

def dcg_at_k(recs, truths, k):
    return sum(1 / np.log2(i + 2) for i, item in enumerate(recs[:k]) if item in truths)

def ndcg_at_k(recs, truths, k):
    ideal_dcg = dcg_at_k(truths, truths, k)
    return dcg_at_k(recs, truths, k) / ideal_dcg if ideal_dcg > 0 else 0.0

def hhi(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    return sum((count / total) ** 2 for count in counter.values())

def entropy(recommendations):
    counter = Counter(recommendations)
    total = sum(counter.values())
    return -sum((count / total) * np.log2(count / total) for count in counter.values() if count > 0) if total else 0

def gini_index_scores(x):
    x = np.array(x)
    if x.size == 0:
        return 0.0
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-6
    x_sorted = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

# --- Diversify Recommendation ---
def get_recommendations_s3_music_diversify(user_history, candidate_tracks, k=10):
    artist_scores = user_history['artist'].value_counts().to_dict()

    results = []
    for idx, row in candidate_tracks.iterrows():
        artist_score = artist_scores.get(row['artist'], 0)
        pop = row.get('normalized_popularity', 0)
        noise = np.random.normal(0, 0.5)
        bias = idx * 1e-4
        score = artist_score + 0.7 * pop + noise + bias
        results.append((row['track_name'], score))

    topk = sorted(results, key=lambda x: x[1], reverse=True)[:k]
    titles, raw_scores = zip(*topk) if topk else ([], [])

    raw_scores = np.array(raw_scores)
    if np.std(raw_scores) < 1e-3:
        raw_scores += np.random.normal(0, 0.1, size=len(raw_scores))
    raw_scores -= raw_scores.min()
    raw_scores += 1e-6

    return list(titles), raw_scores.tolist()

# --- Evaluation Loop ---
def evaluate_music_diversify(test_data, save_path=results_path):
    user_ids = test_data['userId'].unique()[:10]
    metrics = []
    all_recs = []
    exposure_scores = Counter()

    for uid in user_ids:
        hist = test_data[test_data['userId'] == uid]
        if len(hist) < 6:
            continue

        input_hist = hist.iloc[:5]
        truth_tracks = hist.iloc[5:]['track_name'].tolist()

        candidate_tracks = test_data[~test_data['track_id'].isin(input_hist['track_id'])].copy()
        track_pop = test_data['track_id'].value_counts()
        candidate_tracks['popularity'] = candidate_tracks['track_id'].map(track_pop).fillna(0)
        candidate_tracks['normalized_popularity'] = candidate_tracks['popularity'] / candidate_tracks['popularity'].max()
        candidate_tracks = candidate_tracks.sample(min(len(candidate_tracks), 120), random_state=42)

        recs, scores = get_recommendations_s3_music_diversify(input_hist, candidate_tracks)

        if not recs:
            continue

        for i, title in enumerate(recs):
            exposure_scores[title] += 1.0 - 0.05 * i

        all_recs.extend(recs)

        metrics.append({
            "user": uid,
            "hit_rate": hit_rate(recs, truth_tracks),
            "avg_rank": average_rank(recs, truth_tracks),
            "recall@5": recall_at_k(recs, truth_tracks, k=5),
            "ndcg@5": ndcg_at_k(recs, truth_tracks, k=5),
            "hhi": hhi(recs),
            "entropy": entropy(recs),
            "gini": gini_index_scores(scores)
        })

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path, index=False)

    print("\nAverage Metrics (S3 Music Diversify):")
    print(df_metrics.mean(numeric_only=True))

    print("\nSystem-Level Entropy:", entropy(all_recs))
    print("System-Level Gini Index:", gini_index_scores(list(exposure_scores.values())))
