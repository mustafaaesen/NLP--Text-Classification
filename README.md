# NLP & Text Classification Projects 📊🧠

This repository contains multiple natural language processing (NLP) projects focused on sentiment analysis and text classification using both traditional machine learning techniques (TF-IDF + Naive Bayes) and advanced transformer-based models (BERT). The datasets and implementations are entirely in Turkish.

---

## 📁 Project Structure

NLP &Text Classification/
├── NLP_&Text_Classification.ipynb
├── nlp_&text_classification.py
│
├── NLP Bert Model/
│ ├── NLP_News_Category_Project_2.ipynb
│ └── nlp_news_category_project_2.py
│
└── NLP TF-IDF Model/
├── NLP_News_Category_Project_1.ipynb
├── nlp_news_category_project_1.py
├── NLP_News_Category_Project_3.ipynb
├── nlp_news_category_project_3.py
├── NLP_News_Category_Project_4.ipynb
└── nlp_news_category_project_4.py


---

## 🧪 Project 1: Sentiment Analysis on Turkish IMDb Reviews (TF-IDF + Naive Bayes)

**Goal:**  
Classify Turkish movie reviews as **positive** or **negative**.

**Pipeline:**
- Data loading and cleaning (lowercase, punctuation removal, digit filtering)
- Stopword removal (Turkish)
- Lemmatization and stemming
- Vectorization using `CountVectorizer` and `TfidfVectorizer`
- Model training using `Multinomial Naive Bayes`
- Model evaluation with accuracy and confusion matrix
- Prediction function for user input
- Visualization using `WordCloud`

**Result:**  
Achieved ~86.5% accuracy on test data.

---

## 🧪 Project 2: News Category Classification with BERT (Multiclass Text Classification)

**Goal:**  
Classify Turkish news headlines into one of 7 categories:
- Ekonomi, Magazin, Sağlık, Siyaset, Spor, Teknoloji, Yaşam

**Pipeline:**
- Dataset analysis and label encoding
- Data tokenization using `transformers` (DistilBERT)
- Fine-tuning a pretrained BERT model
- Evaluation using `Trainer` and classification report
- Input function for real-time prediction

**Result:**  
High accuracy and balanced performance across all classes.

---
## 📁 Datasets
- `data/turkish_imdb_dataset.csv`: Sentiment analysis dataset for Turkish IMDb movie reviews.
- `data/turkish_news_dataset.csv`: News headline classification dataset with 7 categories.


## 🧠 Key Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn, WordCloud
- NLTK, Zemberek (optional)
- Hugging Face Transformers (BERT)
- Google Colab / Jupyter Notebook

---

## ⚙️ How to Run

1. Clone the repo:
```bash
git clone https://github.com/mustafaaesen/nlp-text-classification.git
cd nlp-text-classification
