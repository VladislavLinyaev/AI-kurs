
import re
import random
from datetime import datetime

from weather_api import get_weather, get_weather_forecast
from database import init_db, save_user, log_weather_query
from dialog_manager import dialog_manager, DialogState
from intent_classifier_embeddings import intent_classifier
from logger import log_message
from extractors import extract_city, extract_date_offset, is_weather_query

class ChatBot:
    def __init__(self):
        self.name = None
        self.current_user_id = None
        self.waiting_for_name = False
        init_db()
    
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
        return f"Приятно познакомиться, {self.name}!"

    def handle_addition(self, text):
       
        # Проверяем формат 2+2
        match = re.search(r'(\d+)\s*\+\s*(\d+)', text)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат сложения: {a} + {b} = {a + b}"
        
        # Проверяем текстовый формат "два плюс два"
        numbers = {
            'ноль': 0, 'один': 1, 'два': 2, 'три': 3, 'четыре': 4,
            'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
            'десять': 10
        }
        
        text_lower = text.lower()
        if 'плюс' in text_lower:
            for word, num in numbers.items():
                if word in text_lower:
                    # Ищем второе число
                    for word2, num2 in numbers.items():
                        if word2 in text_lower and word != word2:
                            return f"Результат сложения: {num} + {num2} = {num + num2}"
        
        return "Скажите, например: 2+2 или два плюс два"

    def handle_subtraction(self, text):
        
        match = re.search(r'(\d+)\s*-\s*(\d+)', text)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат вычитания: {a} - {b} = {a - b}"
        return None

    def handle_multiplication(self, text):
       
        match = re.search(r'(\d+)\s*\*\s*(\d+)', text)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат умножения: {a} * {b} = {a * b}"
        return None

    def handle_division(self, text):
        
        match = re.search(r'(\d+)\s*/\s*(\d+)', text)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            if b != 0:
                return f"Результат деления: {a} / {b} = {a / b}"
            return "На ноль делить нельзя!"
        return None

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

    def handle_weather(self, message):
        city = extract_city(message)
        if not city:
            if self.current_user_id:
                dialog_manager.set_state(self.current_user_id, DialogState.WAIT_CITY)
                return "В каком городе вас интересует погода?"
            return "Пожалуйста, представьтесь сначала."
        
        offset, day_label = extract_date_offset(message)
        if self.current_user_id:
            log_weather_query(self.current_user_id, city)
        
        if offset == 0:
            return get_weather(city)
        else:
            return get_weather_forecast(city, offset, day_label)
    
    def handle_weather_with_state(self, message):
        if not self.current_user_id:
            return None
        
        user_id = self.current_user_id
        state = dialog_manager.get_state(user_id)
        
        if state == DialogState.START:
            if is_weather_query(message):
                city = extract_city(message)
                if city:
                    offset, day_label = extract_date_offset(message)
                    log_weather_query(user_id, city)
                    return get_weather(city) if offset == 0 else get_weather_forecast(city, offset, day_label)
                else:
                    dialog_manager.set_state(user_id, DialogState.WAIT_CITY)
                    return "В каком городе вас интересует погода?"
            return None
        
        elif state == DialogState.WAIT_CITY:
            city = extract_city(f"в {message}")
            if not city:
                city = message.strip().capitalize()
            log_weather_query(user_id, city)
            result = get_weather(city)
            dialog_manager.reset(user_id)
            return result
        
        return None

    def process(self, message):
        message_clean = message.strip()
        
        
        if self.waiting_for_name and re.match(r'^[а-яА-ЯёЁa-zA-Z\s]+$', message_clean):
            self.waiting_for_name = False
            
            name_parts = message_clean.split()
            name = name_parts[0].capitalize()
            response = self.set_name(name)
            log_message(message_clean, response)
            return response
        
        
        if not self.current_user_id and not self.waiting_for_name:
            self.waiting_for_name = True
            response = "Здравствуйте! Как вас зовут?"
            log_message(message_clean, response)
            return response
        
      
        if self.current_user_id:
            state_response = self.handle_weather_with_state(message_clean)
            if state_response:
                log_message(message_clean, state_response)
                return state_response
        
        
        if '+' in message_clean or 'плюс' in message_clean:
            response = self.handle_addition(message_clean)
            if "Результат" in response:
                log_message(message_clean, response)
                return response
        
       
        if '-' in message_clean and not any(c.isalpha() for c in message_clean.replace('-', '')):
            response = self.handle_subtraction(message_clean)
            if response:
                log_message(message_clean, response)
                return response
        
        
        if '*' in message_clean:
            response = self.handle_multiplication(message_clean)
            if response:
                log_message(message_clean, response)
                return response
        
        
        if '/' in message_clean:
            response = self.handle_division(message_clean)
            if response:
                log_message(message_clean, response)
                return response
        
        
        if message_clean.lower() in ['привет', 'здравствуй', 'здравствуйте']:
            response = self.greet()
            log_message(message_clean, response)
            return response
        
        if message_clean.lower() in ['пока', 'до свидания', 'выход']:
            response = self.farewell()
            log_message(message_clean, response)
            return response
        
        if 'дела' in message_clean.lower():
            response = self.how_are_you()
            log_message(message_clean, response)
            return response
        
        if 'время' in message_clean.lower() or 'час' in message_clean.lower():
            response = self.what_time()
            log_message(message_clean, response)
            return response
        
        if 'спасиб' in message_clean.lower():
            response = self.thanks()
            log_message(message_clean, response)
            return response
        
        
        intent, confidence = intent_classifier.predict_intent(message_clean)
        print(f"[DEBUG] Intent: {intent}, Confidence: {confidence:.2%}")
        
     
        if intent == "greeting":
            response = self.greet()
        elif intent == "goodbye":
            response = self.farewell()
        elif intent == "weather":
            response = self.handle_weather(message_clean)
        elif intent == "how_are_you":
            response = self.how_are_you()
        elif intent == "time":
            response = self.what_time()
        elif intent == "addition":
            response = self.handle_addition(message_clean)
        elif intent == "ask_name":
            response = self.ask_name()
        elif intent == "thanks":
            response = self.thanks()
        else:
            response = "Я не совсем понял. Могу рассказать о погоде, времени или посчитать примеры (2+2, 5-3)."
        
        log_message(message_clean, response)
        return response

bot = ChatBot()