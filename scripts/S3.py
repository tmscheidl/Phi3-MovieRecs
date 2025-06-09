import os
import re
import gc
import torch
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from fuzzywuzzy import fuzz
from transformers import AutoModelForCausalLM, AutoTokenizer

# Device configuration
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
data_path = os.path.join("data")
results_path = os.path.join("results", "s7_metrics.csv")
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

# --- LLM Recommendation Function ---
def get_recommendations_s7_cot(user_history, df_items, candidate_df, tokenizer, model, device, k=10):
    watched_titles = df_items[df_items['itemId'].isin(user_history['itemId'])]['title'].tolist()
    user_movies_str = ", ".join(watched_titles[:5])

    prompt = (
        f"The user has watched the following movies: {user_movies_str}. "
        f"Please recommend {k} real movie titles that match their taste. "
        f"List them clearly like: 1. Title - Reason"
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

    valid_titles = candidate_df['title'].tolist()
    filtered_titles = [title for title in recommended_titles if title in valid_titles]

    return filtered_titles[:k], decoded

# --- Evaluation Loop ---
user_ids = test_data_ml1m_fullInteraction['userId'].unique()[:10]
metrics_s7 = []
all_recommendations_s7 = []
item_exposure_scores = defaultdict(float)

for uid in user_ids:
    hist = test_data_ml1m_fullInteraction[test_data_ml1m_fullInteraction['userId'] == uid]
    titles_all = df_items_ml[df_items_ml['itemId'].isin(hist['itemId'])]['title'].tolist()
    if len(titles_all) < 6:
        continue

    input_titles = titles_all[:5]
    truth = titles_all[5:]
    candidate_movies_df_s7 = df_items_ml[~df_items_ml['title'].isin(input_titles)].copy()

    pop_series = test_data_ml1m_fullInteraction['itemId'].value_counts()
    candidate_movies_df_s7['popularity'] = candidate_movies_df_s7['itemId'].map(pop_series).fillna(0)
    candidate_movies_df_s7['normalized_popularity'] = candidate_movies_df_s7['popularity'] / candidate_movies_df_s7['popularity'].max()

    recs_titles, full_response = get_recommendations_s7_cot(
        hist, df_items_ml, candidate_movies_df_s7, tokenizer, model, device
    )

    print(f"\nUser {uid}")
    print(f"Ground truth: {truth}")
    print(f"Recommendations: {recs_titles}")

    # Simulated exposure score per rank
    for i, title in enumerate(recs_titles):
        score = 1.0 - 0.05 * i  # Decreasing by rank
        item_exposure_scores[title] += score

    # Accumulate for entropy
    all_recommendations_s7.extend(recs_titles)

    metrics_s7.append({
        "hit_rate": hit_rate(recs_titles, truth),
        "avg_rank": average_rank(recs_titles, truth),
        "recall@5": recall_at_k(recs_titles, truth, k=5),
        "ndcg@5": ndcg_at_k(recs_titles, truth, k=5),
        "hhi": hhi(recs_titles),
        "entropy": entropy_metric(recs_titles)
        # Gini is calculated later globally
    })

# --- Metrics Summary ---
df_metrics_s7 = pd.DataFrame(metrics_s7)
print("\nAverage Metrics for First 10 Users (S7 COT):")
print(df_metrics_s7.mean())

print("\nSystem-Level Entropy Across All S7 Recommendations:")
print(entropy_metric(all_recommendations_s7))

print("\nSystem-Level Gini Index Across All S7 Recommendations:")
gini_scores = list(item_exposure_scores.values())
print(gini_index_scores(gini_scores))

# Save metrics to a CSV file in the results folder
os.makedirs("results", exist_ok=True)
df_metrics_s7.to_csv(results_path, index=True)
print("\nSaved metrics to", results_path)