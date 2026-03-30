# intent_classifier.py
import spacy
import pickle
import os

class IntentClassifier:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Загрузка обученных моделей"""
        if os.path.exists("intent_model.pkl") and os.path.exists("vectorizer.pkl"):
            try:
                with open("intent_model.pkl", "rb") as f:
                    self.model = pickle.load(f)
                with open("vectorizer.pkl", "rb") as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Модель интентов загружена")
                return True
            except Exception as e:
                print(f"❌ Ошибка загрузки модели: {e}")
        else:
            print("⚠️ Файлы модели не найдены!")
        return False
    
    def preprocess(self, text: str) -> str:
        """Предобработка текста"""
        doc = self.nlp(text.lower())
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and not token.is_space and not token.like_num:
                if len(token.lemma_) > 2:
                    tokens.append(token.lemma_)
        return " ".join(tokens)
    
    def predict_intent(self, text: str, threshold: float = 0.5) -> tuple:
        """Предсказание интента с уверенностью"""
        if not self.model or not self.vectorizer:
            return self._fallback(text), 0.3
        
        try:
            processed = self.preprocess(text)
            vector = self.vectorizer.transform([processed])
            
            probabilities = self.model.predict_proba(vector)
            confidence = max(probabilities[0])
            intent = self.model.predict(vector)[0]
            
            if confidence < threshold:
                return "unknown", confidence
            
            return intent, confidence
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return self._fallback(text), 0.3
    
    def _fallback(self, text: str) -> str:
        """Fallback определение интента"""
        text_lower = text.lower()
        if any(w in text_lower for w in ['привет', 'здравствуй', 'добрый']):
            return "greeting"
        elif any(w in text_lower for w in ['пока', 'до свидания']):
            return "goodbye"
        elif any(w in text_lower for w in ['погода', 'дождь', 'температура']):
            return "weather"
        elif any(w in text_lower for w in ['дела', 'настроение']):
            return "how_are_you"
        elif any(w in text_lower for w in ['время', 'час']):
            return "time"
        elif '+' in text_lower:
            return "addition"
        return "unknown"

intent_classifier = IntentClassifier()