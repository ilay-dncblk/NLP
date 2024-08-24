import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import nltk
nltk.download('stopwords')

# Veriyi yükleme
df = pd.read_csv("tweets_labeled.csv")

# Veri ön işleme
df['tweet'] = df['tweet'].astype(str)

# Metin temizleme fonksiyonu
def clean_text(text):
    # Küçük harfe çevirme
    text = text.lower()
    # URL'leri kaldırma
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Noktalama işaretlerini kaldırma
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Rakamları kaldırma
    text = re.sub(r'\d+', '', text)
    # Stop words'leri kaldırma
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_text = [word for word in words if word not in stop_words]
    # Tekrar birleştirme
    text = ' '.join(filtered_text)
    return text

# Metinleri temizleme
df['clean_tweet'] = df['tweet'].apply(clean_text)

# TF-IDF vektörlerini oluşturma
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_tweet'])

# Etiketlerin hazırlanması
y = df['label']

# Veri setinin eğitim ve test olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Modelin tanımlanması ve eğitilmesi
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans metrikleri
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Görselleştirme
plt.figure(figsize=(10, 8))

# Confusion matrix görselleştirme
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt="d", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

# Precision, Recall, F1 Score görselleştirme
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
metrics_data = {"Precision": precision, "Recall": recall, "F1 Score": fscore, "Accuracy": accuracy}
plt.subplot(1, 2, 2)
plt.bar(metrics_data.keys(), metrics_data.values(), color=['blue', 'green', 'red', 'orange'])
plt.title("Performance Metrics")
plt.ylabel("Score")

plt.tight_layout()
plt.show()
