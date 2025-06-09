# LLM-Based Movie Recommender Systems

This repository contains practical work focused on developing and evaluating multiple movie recommender system models using large language models (LLMs). The objective is to explore diverse recommendation strategies—from genre-based filtering to chain-of-thought (COT) explanation-enhanced approaches—aimed at improving accuracy, diversity, and transparency in movie recommendations.

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
│   └── S7.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── Plot.ipynb                # Visualizations of evaluation results
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

Models S6 and S7 utilize the Phi-3.5-mini-instruct LLM to produce text-based reasoning or motivational explanations for their recommendations.

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

The notebook `Plot.ipynb` visualizes the evaluation metrics to compare model performances in terms of trade-offs between accuracy and diversity.

---

## 📊 Results Summary

- 🔍 Relevance:  
  The COT-based model (`S7`) demonstrates improved Hit Rate and NDCG, showing that LLM-generated explanations enhance recommendation quality.

- 🌐 Diversity:  
  Models `S3–S5` increase diversity scores by promoting novel or underrepresented items.

- 💬 Explainability:  
  `S6` and `S7` offer reasoning or motivational narratives, increasing transparency and user trust.

- 📈 System-Level Entropy:  
  Indicates healthy diversity across the user base in advanced models.

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

All theoretical underpinnings, architectural inspirations, and related works are cited in the reference/ folder using BibTeX and formatted citations.

## ✅ Status
✔ All code, models, evaluation data, and visualizations are complete.

✔ Full experimental pipeline and findings are documented.

✔ Models are reproducible and ready for extension or adaptation.
