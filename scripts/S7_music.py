import os
import re
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
data_path = os.path.join("data")
results_path = os.path.join("results", "s7_music_metrics.csv")
model_name = "microsoft/Phi-3.5-mini-instruct"

# --- Gini Index ---
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

# --- Fuzzy filter ---
def fuzzy_filter_titles(recommended_titles, valid_titles, threshold=85):
    filtered = []
    for rec in recommended_titles:
        best_match = max(valid_titles, key=lambda v: fuzz.ratio(v.lower(), rec.lower()), default=None)
        if best_match and fuzz.ratio(best_match.lower(), rec.lower()) >= threshold:
            filtered.append(best_match)
    return list(dict.fromkeys(filtered))  # dedup while preserving order

# --- LLM-based COT Recommendation ---
def get_recommendations_s7_music_cot(user_history, candidate_df, tokenizer, model, device, k=10):
    listened_tracks = user_history['track_name'].tolist()
    user_tracks_str = ", ".join(listened_tracks[:5])

    prompt = (
        f"The user has listened to the following tracks: {user_tracks_str}. "
        f"Please recommend {k} real track titles that match their taste. "
        f"List them clearly like: 1. Track Title - Reason"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.9,
            do_sample=True,
            top_p=0.95
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Parse recommendations
    lines = decoded.split("\n")
    recommended_titles = []
    for line in lines:
        if "." in line:
            try:
                title_reason = line.split(".", 1)[1].strip()
                if "-" in title_reason:
                    title = title_reason.split("-", 1)[0].strip()
                    recommended_titles.append(title)
            except IndexError:
                continue

    valid_titles = candidate_df['track_name'].tolist()
    filtered_titles = fuzzy_filter_titles(recommended_titles, valid_titles)
    return filtered_titles[:k], decoded

# --- Evaluation Loop ---
def evaluate_music_cot(test_data, tokenizer, model, device, save_path=results_path):
    user_ids = test_data['userId'].unique()[:10]
    metrics = []
    all_recs = []
    exposure_scores = defaultdict(float)

    for uid in user_ids:
        hist = test_data[test_data['userId'] == uid]
        if len(hist) < 6:
            continue

        input_hist = hist.iloc[:5]
        truth_tracks = hist.iloc[5:]['track_name'].tolist()
        candidate_df = test_data[~test_data['track_id'].isin(input_hist['track_id'])].copy()

        pop = test_data['track_id'].value_counts()
        candidate_df['popularity'] = candidate_df['track_id'].map(pop).fillna(0)
        candidate_df['normalized_popularity'] = candidate_df['popularity'] / candidate_df['popularity'].max()

        recs, full_response = get_recommendations_s7_music_cot(input_hist, candidate_df, tokenizer, model, device)

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
            "entropy": entropy(recs)
        })

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path, index=False)

    print("\nAverage Metrics (S7 Music COT):")
    print(df_metrics.mean(numeric_only=True))

    print("\nSystem-Level Entropy:", entropy(all_recs))
    print("System-Level Gini Index:", gini_index_scores(list(exposure_scores.values())))
