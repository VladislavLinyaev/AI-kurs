# patterns.py (обновленная версия)
import re
import random
from datetime import datetime, timedelta

import spacy
from spacy.matcher import Matcher

from weather_api import get_weather, get_weather_forecast
from database import init_db, save_user, log_weather_query
from dialog_manager import dialog_manager, DialogState
from intent_classifier import intent_classifier  # Добавляем классификатор
from logger import log_message  # Добавляем для логирования

nlp = spacy.load("ru_core_news_sm")

matcher = Matcher(nlp.vocab)
matcher.add("WEATHER_QUERY", [
    [{"LEMMA": "погода"}, {"LOWER": {"IN": ["в", "во"]}}, {"POS": "PROPN"}],
    [{"LEMMA": "погода"}, {"LOWER": {"IN": ["в", "во"]}}, {"IS_TITLE": True}],
])

DATE_KEYWORDS = {
    "сегодня": 0,
    "завтра": 1,
    "послезавтра": 2,
    "через два дня": 2,
    "через 2 дня": 2,
}

WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "среду": 2, "среда": 2,
    "четверг": 3, "пятницу": 4, "пятница": 4,
    "субботу": 5, "суббота": 5, "воскресенье": 6, "воскресенье": 6,
}

def extract_date_offset(text: str) -> tuple[int, str]:
    text_lower = text.lower()
    for keyword, offset in DATE_KEYWORDS.items():
        if keyword in text_lower:
            labels = {0: "сегодня", 1: "завтра", 2: "послезавтра"}
            return offset, labels.get(offset, f"через {offset} дня")
    today_wd = datetime.now().weekday()
    for word, target_wd in WEEKDAYS.items():
        if word in text_lower:
            offset = (target_wd - today_wd) % 7
            if offset == 0:
                offset = 7
            return offset, word
    return 0, "сегодня"

def extract_city(text: str) -> str | None:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            lemma = ent.root.lemma_.capitalize()
            return lemma
    m = re.search(r"\bв[о]?\s+([А-ЯЁ][а-яёА-ЯЁ\-]+)", text)
    if m:
        word = m.group(1)
        word_doc = nlp(word)
        lemma = word_doc[0].lemma_.capitalize() if word_doc else word
        return lemma
    return None

def is_weather_query(text: str) -> bool:
    doc = nlp(text)
    if matcher(doc):
        return True
    return "погода" in text.lower()

class ChatBot:
    def __init__(self):
        self.name = None
        self.current_user_id = None
        self.waiting_for_name = False
        self.patterns = []
        init_db()
        self.register_patterns()

    def register_patterns(self):
        # Старые паттерны для обратной совместимости
        self.patterns.append(
            (re.compile(r"(\d+)\s*\+\s*(\d+)"), self.handle_addition)
        )

    def greet(self):
        if self.name:
            return f"Здравствуйте, {self.name}! Чем могу помочь?"
        self.waiting_for_name = True
        return "Здравствуйте! Чем могу помочь? Как вас зовут?"

    def farewell(self):
        if self.name:
            return f"До свидания, {self.name}! Было приятно пообщаться."
        return "До свидания!"

    def set_name(self, name):
        self.name = name.capitalize()
        self.current_user_id = save_user(self.name)
        if self.current_user_id:
            dialog_manager.reset(self.current_user_id)
        return f"Приятно познакомиться, {self.name}!"

    def handle_addition(self, match):
        try:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат сложения: {a} + {b} = {a + b}"
        except Exception:
            return "Не удалось выполнить сложение"

    def how_are_you(self):
        return random.choice([
            "Всё отлично, спасибо!",
            "Хорошо, а у вас?",
            "Прекрасно! Готов помочь.",
        ])

    def what_time(self):
        return f"Сейчас {datetime.now().strftime('%H:%M')}"

    def ask_name(self):
        return "Меня зовут Бот-помощник. А как вас зовут?"

    def thanks(self):
        return random.choice([
            "Пожалуйста! Всегда рад помочь.",
            "Обращайтесь!",
            "Не за что!"
        ])

    def unknown_intent(self):
        return "Я не совсем понял. Могу рассказать о погоде, времени или просто поболтать."

    def handle_weather_with_state(self, message: str) -> str:
        """Обработка погоды с учетом состояния диалога"""
        if not self.current_user_id:
            return "Пожалуйста, представьтесь сначала (например: 'меня зовут Иван')"
        
        user_id = self.current_user_id
        state = dialog_manager.get_state(user_id)
        data = dialog_manager.get_data(user_id)
        
        # Состояние START - ищем запрос погоды
        if state == DialogState.START:
            if is_weather_query(message):
                city = extract_city(message)
                if city:
                    offset, day_label = extract_date_offset(message)
                    if user_id:
                        log_weather_query(user_id, city)
                    if offset == 0:
                        return get_weather(city)
                    else:
                        return get_weather_forecast(city, offset, day_label)
                else:
                    dialog_manager.set_state(user_id, DialogState.WAIT_CITY)
                    return "В каком городе вас интересует погода?"
            return None
        
        # Состояние WAIT_CITY - ожидаем город
        elif state == DialogState.WAIT_CITY:
            city = extract_city(f"в {message}")
            if not city:
                city = message.strip().capitalize()
            
            if user_id:
                log_weather_query(user_id, city)
            
            offset, day_label = extract_date_offset(message)
            result = get_weather(city) if offset == 0 else get_weather_forecast(city, offset, day_label)
            dialog_manager.reset(user_id)
            return result
        
        return None

    def process_with_ml(self, message: str) -> str:
        """Основной метод обработки с использованием ML"""
        message_clean = message.strip()
        
        # Обработка имени, если бот ждет его
        if self.waiting_for_name and re.match(r'^[а-яА-ЯёЁa-zA-Z]+$', message_clean):
            self.waiting_for_name = False
            response = self.set_name(message_clean)
            log_message(message_clean, response)
            return response
        
        # Проверяем состояние диалога (для погоды)
        if self.current_user_id:
            state_response = self.handle_weather_with_state(message_clean)
            if state_response:
                log_message(message_clean, state_response)
                return state_response
        
        # Определяем интент с помощью ML
        intent, confidence = intent_classifier.predict_intent(message_clean)
        
        print(f"[DEBUG] Intent: {intent}, Confidence: {confidence:.2f}")
        
        # Если уверенность низкая
        if confidence < 0.5:
            response = "Я не уверен, что понял запрос. Попробуйте переформулировать."
            log_message(message_clean, response)
            return response
        
        # Обработка в зависимости от интента
        if intent == "greeting":
            response = self.greet()
        elif intent == "goodbye":
            response = self.farewell()
        elif intent == "weather":
            # Проверяем, есть ли город в запросе
            city = extract_city(message_clean)
            if city and self.current_user_id:
                offset, day_label = extract_date_offset(message_clean)
                log_weather_query(self.current_user_id, city)
                response = get_weather(city) if offset == 0 else get_weather_forecast(city, offset, day_label)
            elif self.current_user_id:
                dialog_manager.set_state(self.current_user_id, DialogState.WAIT_CITY)
                response = "В каком городе вас интересует погода?"
            else:
                response = "Пожалуйста, представьтесь сначала."
        elif intent == "how_are_you":
            response = self.how_are_you()
        elif intent == "time":
            response = self.what_time()
        elif intent == "addition":
            # Ищем числа в тексте
            match = re.search(r'(\d+)\s*\+\s*(\d+)', message_clean)
            if match:
                response = self.handle_addition(match)
            else:
                response = "Скажите, например: 5+3"
        elif intent == "ask_name":
            response = self.ask_name()
        elif intent == "thanks":
            response = self.thanks()
        else:
            response = self.unknown_intent()
        
        # Логируем ответ
        if self.current_user_id:
            log_message(message_clean, response)
        
        return response

    def process(self, message: str) -> str:
        """Основной метод для обратной совместимости"""
        return self.process_with_ml(message)

bot = ChatBot()