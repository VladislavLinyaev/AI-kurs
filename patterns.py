
import re
from datetime import datetime

from weather_api import get_weather, get_weather_forecast
from database import init_db, save_user, log_weather_query
from dialog_manager import dialog_manager, DialogState
from intent_classifier import intent_classifier
from skill_router import SkillRouter
from logger import log_message
from extractors import extract_city, extract_date_offset, is_weather_query
from tts_module import speak_async

class ChatBot:
    def __init__(self):
        self.name = None
        self.current_user_id = None
        self.waiting_for_name = False
        init_db()
        
      
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

    def handle_addition(self, text, user_id=None):
        text = text.lower()
        num_map = {
            'ноль': 0, 'один': 1, 'два': 2, 'три': 3, 'четыре': 4,
            'пять': 5, 'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10
        }
        
       
        found_nums = []
        words = re.findall(r'\w+', text)
        
        for word in words:
            if word.isdigit():
                found_nums.append(int(word))
            elif word in num_map:
                found_nums.append(num_map[word])
        
        if len(found_nums) >= 2:
            result = sum(found_nums)
            res_str = f"Результат: {' + '.join(map(str, found_nums))} = {result}"
            return res_str
        
        return "Я могу складывать числа, например: '2+2'."

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
        
        if self.waiting_for_name:
            self.waiting_for_name = False
            
            match = re.search(r'(?:зовут|имя|это)\s+([А-ЯЁа-яё]+)', message_clean, re.I)
            if match:
                name = match.group(1).capitalize()
            else:
                name = message_clean.split()[-1].strip(".,!?").capitalize()
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
        
      
        intent, confidence = intent_classifier.predict_intent(message_clean)
        print(f"[DEBUG BERT] Intent: {intent}, Confidence: {confidence:.2%}")
        
        
        response = self.skill_router.route(
            intent=intent,
            text=message_clean,
            user_id=self.current_user_id
        )
        
        log_message(message_clean, response)
        
        return response

bot = ChatBot()