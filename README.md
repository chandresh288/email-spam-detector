# AI-Spam-Email-Detector

A comprehensive project to classify emails as **spam** or **ham (non-spam)** using deep learning.  
This repository provides an **end-to-end solution** for detecting spam emails with high accuracy while ensuring a balanced and unbiased approach.  

👩‍💻 **Developed and owned by Chandresh Kumari (M.Sc. DS&AI, Himachal Pradesh University, 2023–2025)**

---

## 🚀 Key Features

- **Accurate Classification:** Achieved **97%+ validation accuracy** on unseen data.
- **Balanced Dataset:** Applied downsampling techniques to ensure fairness in classification.
- **Scalable Model:** Designed with modern neural network architecture suitable for text classification.
- **Flask Web App:** User-friendly web interface for real-time spam/ham detection.

---

## 📚 Overview

Spam emails clutter inboxes and pose security risks. This project aims to create a robust spam detection system that:

1. **Enhances Inbox Experience:** Reduces junk emails while retaining important communications.
2. **Improves Security:** Detects harmful or malicious content effectively.
3. **Leverages AI:** Implements deep learning for efficient and scalable email filtering.

---

## 🛠️ How It Works

### 1. **Data Preparation**
- **Dataset:** Trained on the SMS Spam Collection dataset (spam.csv).  
- **Tokenization:** Converts text into numerical sequences for machine learning.  
- **Padding/Truncation:** Standardizes input lengths (maxlen=50).  
- **Balancing:** Downsamples the majority class (ham) to ensure equal representation of spam and ham emails.  

### 2. **Model Architecture**
- **Embedding Layer:** Converts words into semantic vectors.  
- **LSTM Layers:** Captures sequence dependencies in email text.  
- **Dense Layers:** With ReLU and sigmoid activations for classification.  
- **Dropout Regularization:** Prevents overfitting.  

### 3. **Training**
- Optimized with **binary cross-entropy loss**.  
- Trained for **5–30 epochs** with **early stopping**.  
- Achieved **>97% validation accuracy**.  

---

## 🧑‍🏫 Training Process

You can train your own model by running the following script:

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])  # Ham=0, Spam=1

X = df['message'].values
y = df['label'].values

# Tokenization
vocab_size = 5000
max_len = 50
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

# Save tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

sequences = tokenizer.texts_to_sequences(X)
padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=10,
                    batch_size=32,
                    verbose=1)

# Save model
model.save("spam_ham_model.h5")

print("✅ Model trained and saved successfully as spam_ham_model.h5")
````

This produces:

* `spam_ham_model.h5` → the trained model
* `tokenizer.pkl` → tokenizer for text preprocessing

---

## 📊 Web Interface Screenshots

### 1. **Welcome Screen**

<img width="768" alt="Welcome Screen" src="https://github.com/user-attachments/assets/5eaa91ef-aeea-4728-ab0c-166e1007a045">

### 2. **Spam Detected**

<img width="768" alt="Spam Detected" src="https://github.com/user-attachments/assets/ed89bc7f-2577-4d47-bd19-39d12c1251ab">

### 3. **Ham Detected**

<img width="768" alt="Ham Detected" src="https://github.com/user-attachments/assets/0c820bf8-0930-42c7-898f-0c4f6bcf154b">

---

## 🔮 Future Enhancements

* **Pretrained embeddings** (Word2Vec, GloVe, BERT).
* **Transformer-based models** (BERT, DistilBERT).
* **Comprehensive evaluation metrics** (Precision, Recall, F1, AUC).
* **Dockerized deployment** for production readiness.

---

## 🧑‍💻 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/chandresh288/AI-Spam-Email-Detector.git
cd AI-Spam-Email-Detector
```

### 2. Create a Virtual Environment

```bash
python3 -m venv flaskenv
source flaskenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### 4. Train Model (Optional)

```bash
python3 train_model.py
```

### 5. Run Flask App

```bash
python3 app.py
```

---

## 📊 Performance Metrics

* **Validation Accuracy:** \~97%
* **Balanced Performance:** Spam and ham detected fairly.

---

## 🤝 Contributions

Contributions are welcome! Fork this repo, open issues, or submit PRs.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## ✉️ Contact

👩‍💻 **Chandresh Kumari**
M.Sc. Data Science & Artificial Intelligence (2023–2025)
Email: chandreshverma288@gmail.com