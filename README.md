# LLM-Based Movie Recommender Systems

This repository contains practical work focusing on the development and evaluation of models for several movie recommender systems using large language models (LLMs). The aim is to explore different recommender strategies - from genre-based filtering to chain-of-thought (COT) approaches with annotation - and to compare these different models in terms of accuracy, diversity and transparency in movie recommendations.

---

## 📁 Repository Structure

```text
├── scripts/                       # Python scripts for each model (S1 to S7)
│   ├── S1.py
│   ├── S2.py
│   ├── S3.py
│   ├── S4.py
│   ├── S5.py
│   ├── S6.py
│   ├── S7.py
│   ├── S3_music.py
│   └── S7_music.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── Plot_movie.ipynb         # Visualizations of evaluation results
│   ├── Plot_music.ipynb  
│   └── Movie_rec.ipynb           # Full pipeline: loading, evaluation, metrics
│
├── results/                      # Evaluation outputs
│   ├── Evaluation_Metrics_Table.csv
│   ├── Recall_and_NDCG_Results.csv
│   └── System-Level_Entropy.csv
│
├── data/                         # Preprocessed MovieLens data
├── model/                        # Model architecture and configuration
├── reference/                    # Academic references and citations
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview and documentation

```

---

## 🧠 Methodology

### 1. Data Preparation

User profiles, item metadata, and user-movie interaction histories are extracted from the MovieLens dataset. These are processed to generate user histories and candidate movie pools.

### 2. Model Implementations

Each Python script in the `scripts/` folder represents a recommendation model:

- `S1`: Baseline recommendation (e.g., popularity-based)  
- `S2`: Strict genre-matching recommendations  
- `S3`: Diversification-focused recommender  
- `S4`: Quality-aware diversification  
- `S5`: Surprise-enhanced recommender  
- `S6`: Motivation reasoning using LLM  
- `S7`: Chain-of-Thought (COT) reasoning with LLM  
- `S3_music`: Adaptation of the diversification-focused recommender for the **LastFM-1K dataset** (`test_data_lastFM1k_fullInteraction_80users.csv`)  
- `S7_music`: Adaptation of the Chain-of-Thought (COT) reasoning recommender for the **LastFM-1K dataset** (`test_data_lastFM1k_fullInteraction_80users.csv`)  

Models `S6`, `S7`, and `S7_music` utilize the **Phi-3.5-mini-instruct** LLM to generate text-based reasoning or motivational explanations for their recommendations.

### 3. Evaluation Metrics

Models are assessed using both relevance- and diversity-oriented metrics:

- Relevance:  
  - Hit Rate  
  - Average Rank  
  - Recall@5  
  - NDCG@5

- Diversity:  
  - Herfindahl-Hirschman Index (HHI)  
  - Entropy  
  - Gini Index

- System-Level Metrics:  
  - Entropy across recommendations for all users

### 4. Visualization and Analysis

The notebooks `Plot_movie.ipynb` and `Plot_music.ipynb` visualize evaluation metrics using a variety of plots to compare model performances.

#### Movie Plots
Includes:

- Normalized Metric Comparison Across Models (bar chart)  
- Recall@5 and NDCG@5 comparison (bar chart)  
- Diversity metrics (HHI, Entropy, Gini, System-Level Entropy) across models (bar chart)  
- Radar plots for individual models and all models together  
- Line plot of normalized metrics by model  
- Heatmap of model performance across users and metrics  
- Scatter plot showing Accuracy vs Diversity  
- Horizontal bar chart for System-Level Entropy  

#### Music Plots (S3 vs S7 on LastFM-1K dataset)
Includes:

- **Bar chart of average metrics** (Hit Rate, Avg Rank, Recall@5, NDCG@5, HHI, Entropy, Gini) comparing `S3 Diversify` vs `S7 COT`  
- **System-level comparison** of Entropy and Gini between the two models  
- **Radar chart** of normalized metrics to show trade-offs across dimensions  
- **Relative improvement plot** (% improvement of `S7` over `S3` for each metric)  
- **Scatter plot** showing the Accuracy vs Diversity trade-off (`NDCG@5` vs `Entropy`)  

These plots provide insight into how diversification (S3) and reasoning-based recommendations (S7) behave differently on the **LastFM-1K dataset** (`test_data_lastFM1k_fullInteraction_80users.csv`).

## 📊 Results Summary

### Accuracy & Relevance

- S1 (Simple) achieves perfect Hit Rate and leads in Recall@5 and NDCG@5, indicating strong top-5 precision. However, it suffers from low diversity, often recommending the same popular items.
- S2 (Genre-Focused) and S3 (Diversify + xLSTM) maintain moderate hit rates, but lower recall and NDCG suggest many irrelevant top recommendations.
- S7 (Chain-of-Thought) stands out for ranking quality (Avg. Rank = 2.10), showing its ability to surface relevant items higher, but Recall remains low, indicating room to improve overall relevance coverage.

### Diversity & Novelty

- S5 (Surprise) and S6 (Motivate Reasoning) strike a balance between novelty and diversity, with high system-level entropy and moderate Gini Index. They introduce more obscure but interesting items.
- S3 (Diversify) shows signs of over-personalization (Gini = 0.4427), likely reinforcing niche user preferences too narrowly.
- S7 (COT) presents promising diversity, balancing between relevance and novelty, though coverage and recall need optimization.

 ### Explainability & Transparency

- S6 and S7 lead in explainability, producing natural-language rationales that increase user trust and understanding.
- Models S5–S7 prioritize human-centered transparency, making them well-suited for trust-aware applications, even at some cost to predictive accuracy.

---

## ▶️ Usage

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Run any recommendation model:

```bash
python scripts/S1.py   # Replace S1 with any model from S1 to S7
```
3. View or analyze the results:

The results will be stored in the results/ folder as CSV files:

There are some previous results:

- Evaluation_Metrics_Table.csv

- Recall_and_NDCG_Results.csv

- System-Level_Entropy.csv

You can check the visualisation of the performance using by the notebook:

```bash
jupyter notebook notebooks/Plot.ipynb
```
## 📚 References

This practical work is inspired by the GitHub repository [Benchmark_RecLLM_Fairness](https://github.com/yasdel/Benchmark_RecLLM_Fairness) and is based on the methodology and findings presented in the paper “Benchmarking Large Language Models as Recommender Systems” (Delbar et al., 2024), available in the `reference/` folder.

## ✅ Status
✔ All code, models, evaluation data, and visualizations are complete.

✔ Full experimental pipeline and findings are documented.

✔ Models are reproducible and ready for extension or adaptation.
