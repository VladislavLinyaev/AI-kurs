# train_embeddings.py
import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("ОБУЧЕНИЕ НА WORD EMBEDDINGS (spaCy)")
print("=" * 60)


print("\n Загрузка spaCy модели с векторами...")
try:
    nlp = spacy.load("ru_core_news_lg")
    print("    Загружена ru_core_news_lg")
except:
    print("    ru_core_news_lg не найдена, загружаем ru_core_news_sm...")
    nlp = spacy.load("ru_core_news_sm")


VECTOR_DIM = nlp.vocab.vectors_length
print(f"    Размерность векторов: {VECTOR_DIM}")
print(f"    Количество векторов: {nlp.vocab.vectors.shape[0]}")

if VECTOR_DIM == 0:
    print(" ОШИБКА: Модель не имеет векторов!")
    print("   Установите: pip install ru-core-news-lg")
    exit(1)

def get_sentence_embedding(text):
   
    doc = nlp(text.lower())
    vectors = []
    
    for token in doc:
        
        if not token.is_stop and not token.is_punct and not token.is_space:
            if token.has_vector:
                vectors.append(token.vector)
    
    if vectors:
       
        return np.mean(vectors, axis=0)
    else:
       
        return np.zeros(VECTOR_DIM)

def main():
    
    print("\n Загрузка датасета...")
    df = pd.read_csv("dataset.csv")
    print(f"    Загружено {len(df)} примеров")
    print(f"    Интенты:\n{df['intent'].value_counts()}")
    
   
    print("\n Получение Word Embeddings...")
    print("    Обработка текстов...")
    
    embeddings_list = []
    for i, text in enumerate(df['text']):
        if (i + 1) % 50 == 0:
            print(f"   Обработано {i + 1}/{len(df)} примеров...")
        
        emb = get_sentence_embedding(text)
        embeddings_list.append(emb)
    
    
    X = np.array(embeddings_list)
    y = df['intent']
    
    print(f"\n    Размерность признаков: {X.shape}")
    print(f"    Длина вектора: {X.shape[1]}")
    
    
    print("\n Разделение на train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {X_train.shape[0]} примеров")
    print(f"    Test: {X_test.shape[0]} примеров")
    
    
    print("\n Обучение логистической регрессии...")
    model = LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=10,
        solver='liblinear'
    )
    model.fit(X_train, y_train)
    print("    Обучение завершено")
    
    
    print("\n Оценка модели:")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred))
    
   
    print("\n Сохранение модели...")
    with open("intent_model_embeddings.pkl", "wb") as f:
        pickle.dump(model, f)
    print("    Модель сохранена в 'intent_model_embeddings.pkl'")
    
  
    print("\n Тестирование на новых фразах:")
    test_phrases = [
        "привет",
        "здравствуйте как дела",
        "какая погода в москве",
        "сколько время",
        "2+2 сколько будет",
        "как тебя зовут",
        "спасибо большое",
        "до свидания",
        "погода завтра",
        "который час"
    ]
    
    for phrase in test_phrases:
        emb = get_sentence_embedding(phrase)
        emb_reshaped = emb.reshape(1, -1)
        
        probs = model.predict_proba(emb_reshaped)
        confidence = max(probs[0])
        intent = model.predict(emb_reshaped)[0]
        
        print(f"   '{phrase}' -> {intent} (уверенность: {confidence:.2%})")
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ НА WORD EMBEDDINGS ЗАВЕРШЕНО!")
    print("=" * 60)

if __name__ == "__main__":
    main()