# Bag of Words (BoW) vs. TF-IDF for Sentiment Analysis

A comparative study evaluating traditional text vectorization techniques—**Bag of Words (BoW)** and **Term Frequency-Inverse Document Frequency (TF-IDF)**—paired with a **Logistic Regression** classifier for binary sentiment analysis on movie reviews.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Vectorization Techniques](#-vectorization-techniques)
- [Experimental Setup](#-experimental-setup)
- [Results & Performance Comparison](#-results--performance-comparison)
- [Key Insights](#-key-insights)
- [Repository Structure](#-repository-structure)
- [Installation & Quickstart](#-installation--quickstart)

---

## 🎯 Overview

In Natural Language Processing (NLP), converting raw text into numeric feature vectors is a critical step prior to classification. This repository benchmarks two fundamental feature extraction pipelines:

1. **Bag of Words (CountVectorizer)**: Counts the absolute frequency of words in each review.
2. **TF-IDF (TfidfVectorizer)**: Evaluates word importance by balancing local term frequency against global inverse document frequency.

Both representations are evaluated using a **Logistic Regression** classifier on identical train/test splits.

---

## 📊 Dataset

- **Dataset**: IMDb Movie Reviews Dataset (`dataset/IMDB Dataset.csv`)
- **Total Samples**: 5,000 reviews
- **Classes**: Binary (Positive / Negative)
- **Split**: 80% Training set (4,000 reviews), 20% Testing set (1,000 reviews) with fixed random seed (`random_state=42`).

---

## ⚙️ Vectorization Techniques

| Feature Scheme | Description | Scikit-Learn Class | Stop Words |
| :--- | :--- | :--- | :--- |
| **Bag of Words (BoW)** | Converts text documents into a matrix of token counts. | `CountVectorizer` | English |
| **TF-IDF** | Scales word counts by how unique or rare they are across the dataset. | `TfidfVectorizer` | English |

---

## 🧪 Experimental Setup

- **Classifier**: Logistic Regression (`max_iter=1000`)
- **Evaluation Metrics**:
  - **Accuracy**: Proportion of overall correct predictions.
  - **Precision**: Ratio of true positive sentiment predictions to total positive predictions.
  - **Recall**: Ratio of true positive sentiment predictions to all actual positive reviews.
  - **F1 Score**: Harmonic mean of Precision and Recall.

---

## 📈 Results & Performance Comparison

Empirical evaluation on the 10,000 test reviews yielded the following results:

| Metric | Bag of Words (BoW) | TF-IDF | Gain with TF-IDF |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 85.90% | **88.10%** | **+2.20%** |
| **Precision** | 84.88% | **86.51%** | **+1.63%** |
| **Recall** | 86.45% | **89.53%** | **+3.08%** |
| **F1 Score** | 85.66% | **87.99%** | **+2.33%** |

---

## 💡 Key Insights

1. **TF-IDF Outperforms BoW**: TF-IDF achieved higher scores across all metrics (+2.20% Accuracy, +2.33% F1 Score).
2. **Downweighting Common Words**: While Bag of Words penalizes uninformative words through stop word removal, TF-IDF further refines feature weights by penalizing terms that appear frequently across all reviews (e.g., "movie", "film", "scene").
3. **Higher Recall**: TF-IDF significantly boosted Recall to 89.53% (+3.08% over BoW), capturing subtle positive sentiment nuances more effectively.

---

## 📁 Repository Structure

```
Bag-of-Words-BoW-and-TF-IDF/
├── dataset/
│   └── IMDB Dataset.csv       # IMDb 5,000 movie reviews dataset
├── models/
│   ├── bow_model.py           # CountVectorizer + Logistic Regression model pipeline
│   └── tfidf_model.py         # TfidfVectorizer + Logistic Regression model pipeline
├── utils.py                   # Data loader & preprocessing utility
├── main.py                    # Main script to execute models & display comparison
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Quickstart

### 1. Clone Repository
```bash
git clone https://github.com/yuva-1237/Bag-of-Words-BoW-and-TF-IDF.git
cd Bag-of-Words-BoW-and-TF-IDF
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments
```bash
python main.py
```
