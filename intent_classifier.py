import spacy
import numpy as np
import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IntentClassifier:
    def __init__(self):
        self.nlp = spacy.load("ru_core_news_sm")
        self.bert_model = None
        self.bert_tokenizer = None
        self.use_bert = False

      
        if os.path.exists("./intent_model"):
            try:
                print("📚 Загрузка BERT модели...")
                self.bert_tokenizer = AutoTokenizer.from_pretrained("./intent_model")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained("./intent_model")
                self.bert_model.eval()
                self.use_bert = True

                
                raw = self.bert_model.config.id2label
                self.bert_id2label = {int(k): v for k, v in raw.items()}

                print(f"   ✅ BERT загружен. Интенты: {list(self.bert_id2label.values())}")
                print(f"   ✅ Количество интентов: {len(self.bert_id2label)}")
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки BERT: {e}")
                self.use_bert = False

       
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
        if any(w in text_lower for w in ['число', 'дата', 'день недели', 'месяц']):
            return "date", 0.8
        if '+' in text_lower or 'плюс' in text_lower:
            return "addition", 0.8
        if any(w in text_lower for w in ['зовут', 'имя', 'как тебя', 'кто ты', 'представься']):
            return "ask_name", 0.8
        if any(w in text_lower for w in ['спасиб', 'благодар']):
            return "thanks", 0.8
        if any(w in text_lower for w in ['помощь', 'помоги', 'умеешь', 'можешь', 'функции']):
            return "help", 0.8

        return "unknown", 0.3

    def predict_intent(self, text: str, threshold: float = 0.5) -> tuple:
       
        try:
            # 1. BERT
            if self.use_bert and self.bert_model is not None:
                inputs = self.bert_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=64
                )
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    confidence, pred = torch.max(probs, dim=1)
                    pred_class = pred.item()          # int
                    conf_val = confidence.item()

                
                if pred_class in self.bert_id2label:
                    intent = self.bert_id2label[pred_class]
                    if conf_val >= threshold:
                        print(f"[BERT] {intent} ({conf_val:.2%})")
                        return intent, conf_val
                else:
                    print(f"⚠️ BERT вернул индекс {pred_class}, не найден в id2label")

            
            if self.model is not None and self.vectorizer is not None:
                doc = self.nlp(text.lower())
                vectors = [
                    t.vector for t in doc
                    if not t.is_stop and not t.is_punct and t.has_vector
                ]
                if vectors:
                    emb = np.mean(vectors, axis=0).reshape(1, -1)
                    probs = self.model.predict_proba(emb)[0]
                    conf_val = float(max(probs))
                    intent = self.model.predict(emb)[0]
                    if conf_val >= threshold:
                        print(f"[Embeddings] {intent} ({conf_val:.2%})")
                        return intent, conf_val

            
            intent, conf_val = self._rule_based_predict(text)
            print(f"[Rules] {intent} ({conf_val:.2%})")
            return intent, conf_val

        except Exception as e:
            print(f"⚠️ Ошибка в predict_intent: {e}")
            return "unknown", 0.3



intent_classifier = IntentClassifier()



def predict_intent(text: str) -> str:
    intent, _ = intent_classifier.predict_intent(text)
    return intent


def predict_with_confidence(text: str) -> tuple:
    return intent_classifier.predict_intent(text)