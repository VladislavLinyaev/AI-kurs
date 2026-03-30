
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print(" Загрузка spaCy...")
nlp = spacy.load("ru_core_news_sm")

def preprocess(text):
    
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
       
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.lemma_)
    return " ".join(tokens)

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 60)
    
    
    print("\n Загрузка датасета...")
    df = pd.read_csv("dataset.csv")
    print(f"    Загружено {len(df)} примеров")
    print(f"    Интенты:\n{df['intent'].value_counts()}")
    
    
    df = df.dropna()
    df = df[df['text'].str.strip() != '']
    
   
    print("\n Предобработка...")
    df['processed'] = df['text'].apply(preprocess)
    
    
    print("\n Векторизация...")
    vectorizer = TfidfVectorizer(
        max_features=3000,  
        ngram_range=(1, 3), 
        min_df=2,
        max_df=0.9
    )
    X = vectorizer.fit_transform(df['processed'])
    y = df['intent']
    print(f"    Признаков: {X.shape[1]}")
    
   
    print("\n Разделение...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
  
    print("\n Обучение...")
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=10,  
        class_weight='balanced',
        solver='liblinear'  
    )
    model.fit(X_train, y_train)
    
    
    print("\n Оценка:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred))
    
   
    print("\n Сохранение...")
    with open("intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("    Модель сохранена")
    
    
    print("\n Тест на новых фразах:")
    test_phrases = [
        "привет",
        "здравствуйте как дела",
        "какая погода в москве",
        "сколько время",
        "2+2 сколько будет",
        "как тебя зовут",
        "спасибо большое",
        "до свидания"
    ]
    
    for phrase in test_phrases:
        processed = preprocess(phrase)
        vector = vectorizer.transform([processed])
        probs = model.predict_proba(vector)
        intent = model.predict(vector)[0]
        confidence = max(probs[0])
        print(f"   '{phrase}' -> {intent} (уверенность: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print(" ГОТОВО!")
    print("=" * 60)

if __name__ == "__main__":
    main()