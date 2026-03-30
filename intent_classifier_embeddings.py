
import spacy
import numpy as np
import pickle
import os

class IntentClassifierEmbeddings:
    def __init__(self):
        print(" Загрузка spaCy для Word Embeddings...")
        
      
        try:
            self.nlp = spacy.load("ru_core_news_lg")
            print("    Загружена ru_core_news_lg")
        except:
            self.nlp = spacy.load("ru_core_news_sm")
            print("    Загружена ru_core_news_sm")
        
        self.VECTOR_DIM = self.nlp.vocab.vectors_length
        print(f"    Размерность векторов: {self.VECTOR_DIM}")
        
        self.model = None
        self.load_model()
    
    def load_model(self):
      
        if os.path.exists("intent_model_embeddings.pkl"):
            try:
                with open("intent_model_embeddings.pkl", "rb") as f:
                    self.model = pickle.load(f)
                print(" Модель на Word Embeddings загружена")
                return True
            except Exception as e:
                print(f" Ошибка загрузки: {e}")
        else:
            print(" Модель не найдена. Сначала запустите train_embeddings.py")
        return False
    
    def get_sentence_embedding(self, text):
        
        doc = self.nlp(text.lower())
        vectors = []
        
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space:
                if token.has_vector:
                    vectors.append(token.vector)
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.VECTOR_DIM)
    
    def predict_intent(self, text, threshold=0.5):
       
        if not self.model:
            return "unknown", 0.0
        
        embedding = self.get_sentence_embedding(text)
        embedding_reshaped = embedding.reshape(1, -1)
        
        probabilities = self.model.predict_proba(embedding_reshaped)
        confidence = max(probabilities[0])
        intent = self.model.predict(embedding_reshaped)[0]
        
        if confidence < threshold:
            return "unknown", confidence
        
        return intent, confidence
    
    def _fallback(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ['привет', 'здравствуй']):
            return "greeting"
        elif any(w in text_lower for w in ['пока', 'до свидания']):
            return "goodbye"
        elif any(w in text_lower for w in ['погода', 'дождь']):
            return "weather"
        return "unknown"

intent_classifier = IntentClassifierEmbeddings()