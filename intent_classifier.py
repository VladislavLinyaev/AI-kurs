import spacy
import numpy as np
import pickle
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IntentClassifier:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.bert_model = None
        self.bert_tokenizer = None
        self.use_bert = False
        
        # Загружаем BERT если есть
        if os.path.exists("./intent_model"):
            try:
                print("📚 Загрузка BERT модели...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained("./intent_model")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained("./intent_model")
                self.bert_model.eval()
                self.use_bert = True
                self.bert_id2label = self.bert_model.config.id2label
                self.bert_num_labels = self.bert_model.config.num_labels
                print(f"   ✅ BERT загружен. Интенты: {list(self.bert_id2label.values())}")
                print(f"   ✅ Количество интентов: {self.bert_num_labels}")
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки BERT: {e}")
                self.use_bert = False
        
        # Загружаем fallback модель
        self.model = None
        self.vectorizer = None
        if os.path.exists("intent_model_embeddings.pkl"):
            try:
                with open("intent_model_embeddings.pkl", "rb") as f:
                    self.model = pickle.load(f)
                with open("vectorizer.pkl", "rb") as f:
                    self.vectorizer = pickle.load(f)
                print("✅ Fallback модель загружена")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки fallback: {e}")
    
    def _rule_based_predict(self, text: str) -> tuple:
        """Правила для fallback - всегда возвращает кортеж (intent, confidence)"""
        text_lower = text.lower()
        
        if any(w in text_lower for w in ['привет', 'здравствуй', 'здравствуйте', 'добрый']):
            return "greeting", 0.8
        if any(w in text_lower for w in ['пока', 'до свидания', 'прощай', 'увидимся']):
            return "goodbye", 0.8
        if any(w in text_lower for w in ['погода', 'дождь', 'снег', 'температура', 'градус', 'прогноз']):
            return "weather", 0.8
        if any(w in text_lower for w in ['дела', 'настроение', 'жизнь', 'поживаешь']):
            return "how_are_you", 0.8
        if any(w in text_lower for w in ['время', 'час', 'который']):
            return "time", 0.8
        if '+' in text_lower or 'плюс' in text_lower:
            return "addition", 0.8
        if any(w in text_lower for w in ['зовут', 'имя', 'как тебя']):
            return "ask_name", 0.8
        if any(w in text_lower for w in ['спасиб', 'благодар']):
            return "thanks", 0.8
        if any(w in text_lower for w in ['помощь', 'помоги', 'умеешь', 'можешь']):
            return "help", 0.8
        
        return "unknown", 0.3
    
    def predict_intent(self, text: str, threshold: float = 0.5) -> tuple:
        """
        Предсказание интента с защитой от ошибок
        Всегда возвращает (intent, confidence)
        """
        print(f"[DEBUG BERT] Обработка: '{text}'")  # ДОБАВЬТЕ ЭТУ СТРОКУ
        try:
            # Сначала пробуем BERT
            if self.use_bert and self.bert_model is not None:
                print(f"[DEBUG BERT] 🔵 ИСПОЛЬЗУЮ BERT")
                inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    confidence, pred = torch.max(probs, dim=1)
                    pred_class = pred.item()
                    
                    # Проверяем, что индекс существует в маппинге
                    if str(pred_class) in self.bert_id2label:
                        intent = self.bert_id2label[str(pred_class)]
                    else:
                        # Если индекса нет - пробуем взять по числовому индексу из списка
                        intent_list = list(self.bert_id2label.values())
                        if pred_class < len(intent_list):
                            intent = intent_list[pred_class]
                        else:
                            # Если всё плохо - fallback
                            print(f"⚠️ BERT вернул индекс {pred_class}, вне диапазона (0-{len(intent_list)-1})")
                            return self._rule_based_predict(text)
                    
                    if confidence.item() >= threshold:
                        return intent, confidence.item()
            
            # Fallback через эмбеддинги
            if self.model is not None and self.vectorizer is not None:
                doc = self.nlp(text.lower())
                vectors = [t.vector for t in doc if not t.is_stop and not t.is_punct and t.has_vector]
                if vectors:
                    emb = np.mean(vectors, axis=0).reshape(1, -1)
                    probs = self.model.predict_proba(emb)[0]
                    confidence = max(probs)
                    intent = self.model.predict(emb)[0]
                    if confidence >= threshold:
                        return intent, confidence
            
            # Финальный fallback - правила
            return self._rule_based_predict(text)
            
        except Exception as e:
            # Если что-то пошло совсем не так - возвращаем unknown
            print(f"⚠️ Ошибка в predict_intent: {e}")
            return "unknown", 0.3

# Создаем глобальный экземпляр
intent_classifier = IntentClassifier()

# Для обратной совместимости
def predict_intent(text: str) -> str:
    intent, _ = intent_classifier.predict_intent(text)
    return intent

def predict_with_confidence(text: str) -> tuple:
    return intent_classifier.predict_intent(text)