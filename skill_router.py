# skill_router.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
from datetime import datetime
import random

class SkillRouter:
    def __init__(self, weather_func, addition_func):
        self.weather_func = weather_func
        self.addition_func = addition_func
        self.last_intent = {}
    
    def time_skill(self) -> str:
        return f"Сейчас {datetime.now().strftime('%H:%M')}"
    
    def date_skill(self) -> str:
        return f"Сегодня {datetime.now().strftime('%d.%m.%Y')}"
    
    def greeting_skill(self) -> str:
        return random.choice(["Здравствуйте!", "Привет!", "Добрый день!"])
    
    def goodbye_skill(self) -> str:
        return random.choice(["До свидания!", "Пока!", "Всего хорошего!"])
    
    def help_skill(self) -> str:
        return """Я умею:
🌤️ Погода - спроси "погода в Москве"
⏰ Время - спроси "сколько время"
📅 Дата - спроси "какое сегодня число"
➕ Сложение - напиши "2+2"
💬 Поболтать - спроси "как дела"
❓ О себе - спроси "кто ты" или "как тебя зовут" """
    
    def smalltalk_skill(self) -> str:
        return random.choice([
            "Всё отлично! А у вас?",
            "Хорошо! Чем могу помочь?",
            "Прекрасно! Как ваши дела?"
        ])
    
    def thanks_skill(self) -> str:
        return random.choice([
            "Пожалуйста!", "Обращайтесь!", "Всегда рад помочь!"
        ])
    
    def ask_name_skill(self) -> str:
        return "Меня зовут Бот-помощник. А как вас зовут?"
    
    def addition_skill(self, text: str) -> str:
        import re
        match = re.search(r'(\d+)\s*\+\s*(\d+)', text)
        if match:
            a, b = float(match.group(1)), float(match.group(2))
            return f"Результат сложения: {a} + {b} = {a + b}"
        return "Скажите, например: 2+2"
    
    def fallback_skill(self) -> str:
        return "Я не понял. Скажите 'помощь' чтобы узнать что я умею."
    
    def route(self, intent: str, text: str, user_id: int = None) -> str:
        if user_id and intent != "conversation":
            self.last_intent[user_id] = intent
        
        # ИСПРАВЛЕННАЯ МАРШРУТИЗАЦИЯ
        if intent == "weather":
            return self.weather_func(text)
        elif intent == "time":
            return self.time_skill()
        elif intent == "date":
            return self.date_skill()
        elif intent == "greeting":
            return self.greeting_skill()
        elif intent == "goodbye":
            return self.goodbye_skill()
        elif intent == "help":
            return self.help_skill()
        elif intent == "how_are_you":
            return self.smalltalk_skill()
        elif intent == "smalltalk":
            return self.smalltalk_skill()
        elif intent == "thanks":
            return self.thanks_skill()
        elif intent == "ask_name":
            return self.ask_name_skill()
        elif intent == "addition":
            return self.addition_skill(text)
        elif intent == "conversation":
            prev = self.last_intent.get(user_id, "help")
            if prev == "help":
                return self.help_skill()
            elif prev == "weather":
                return "Какой город вас интересует?"
            elif prev == "time":
                return self.time_skill()
            elif prev == "date":
                return self.date_skill()
            else:
                return self.help_skill()
        else:
            return self.fallback_skill()

skill_router = None