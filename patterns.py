# patterns.py (обновленная версия с Skill Router)
import re
from datetime import datetime

from weather_api import get_weather, get_weather_forecast
from database import init_db, save_user, log_weather_query
from dialog_manager import dialog_manager, DialogState
from intent_classifier import intent_classifier
from skill_router import SkillRouter
from logger import log_message
from extractors import extract_city, extract_date_offset, is_weather_query

class ChatBot:
    def __init__(self):
        self.name = None
        self.current_user_id = None
        self.waiting_for_name = False
        init_db()
        
        # Инициализируем Skill Router
        self.skill_router = SkillRouter(
            weather_func=self.handle_weather,
            addition_func=self.handle_addition
        )
    
    def greet(self):
        if self.name:
            return f"Здравствуйте, {self.name}! Чем могу помочь?"
        self.waiting_for_name = True
        return "Здравствуйте! Как вас зовут?"

    def farewell(self):
        if self.name:
            return f"До свидания, {self.name}! Было приятно пообщаться."
        return "До свидания!"

    def set_name(self, name):
        self.name = name.capitalize()
        self.current_user_id = save_user(self.name)
        return f"Приятно познакомиться, {self.name}!"

    def handle_addition(self, text):
        match = re.search(r'(\d+)\s*\+\s*(\d+)', text)
        if match:
            a = float(match.group(1))
            b = float(match.group(2))
            return f"Результат сложения: {a} + {b} = {a + b}"
        
        numbers = {
            'ноль': 0, 'один': 1, 'два': 2, 'три': 3, 'четыре': 4,
            'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9,
            'десять': 10
        }
        
        text_lower = text.lower()
        if 'плюс' in text_lower:
            for word, num in numbers.items():
                if word in text_lower:
                    for word2, num2 in numbers.items():
                        if word2 in text_lower and word != word2:
                            return f"Результат сложения: {num} + {num2} = {num + num2}"
        
        return "Скажите, например: 2+2 или два плюс два"

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
        
        # Обработка имени
        if self.waiting_for_name and re.match(r'^[а-яА-ЯёЁa-zA-Z\s]+$', message_clean):
            self.waiting_for_name = False
            name_parts = message_clean.split()
            name = name_parts[0].capitalize()
            response = self.set_name(name)
            log_message(message_clean, response)
            return response
        
        # Если пользователь ещё не представился
        if not self.current_user_id and not self.waiting_for_name:
            self.waiting_for_name = True
            response = "Здравствуйте! Как вас зовут?"
            log_message(message_clean, response)
            return response
        
        # Проверка состояния диалога (погода)
        if self.current_user_id:
            state_response = self.handle_weather_with_state(message_clean)
            if state_response:
                log_message(message_clean, state_response)
                return state_response
        
        # Определяем интент через BERT
        intent, confidence = intent_classifier.predict_intent(message_clean)
        print(f"[DEBUG BERT] Intent: {intent}, Confidence: {confidence:.2%}")
        
        # Маршрутизация через Skill Router
        response = self.skill_router.route(
            intent=intent,
            text=message_clean,
            user_id=self.current_user_id
        )
        
        log_message(message_clean, response)
        return response

bot = ChatBot()