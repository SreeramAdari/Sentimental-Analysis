# 📊 Sentiment Analysis Based on Star Ratings (1–5)

This project performs **multi-class sentiment classification** of text reviews using traditional ML models like **XGBoost, Random Forest**, and **Logistic Regression**. The sentiment is inferred from the review rating and grouped into three classes:

- **Negative** → Ratings 1 & 2
- **Neutral** → Rating 3
- **Positive** → Ratings 4 & 5

---

## 🚀 Features

- Preprocessing of review text (lowercasing, stopwords, lemmatization, etc.)
- Rating → Sentiment label transformation
- TF-IDF feature extraction
- Model training & evaluation (Logistic Regression, Random Forest, XGBoost)
- Confusion matrix and classification report visualization
- Streamlit app for live predictions

---

## 🧠 Models Used

| Model              | Accuracy | Notes                                     |
|-------------------|----------|-------------------------------------------|
| Logistic Regression | ~87%     | Lightweight, good baseline                |
| Random Forest       | ~93%     | Great recall on positive class            |
| XGBoost             | ~92%     | Best balance on precision and recall      |

---

## 📁 Folder Overview

- `data/`: Contains the cleaned review dataset
- `models/`: Saved `.sav` and `.pkl` models
- `notebooks/`: Jupyter notebook for training and evaluation
- `app/`: Contains the `streamlit_app.py` script
- `requirements.txt`: Lists all Python libraries used

---

## 🛠️ Setup

### ✅ Install requirements

```bash
pip install -r requirements.txt
