import re
import string
from datetime import datetime
from patterns import patterns, bot_instance  

def clean_message(message):
    message = message.lower().strip()
    message = message.translate(str.maketrans('', '', string.punctuation))
    return message


def handle_greeting(match=None):
    return "Здравствуйте! Чем могу помочь?"

def handle_farewell(match=None):
    return "До свидания! Было приятно пообщаться."

def handle_weather(match):
    city = match.group(1)
    return f"Погода в городе {city}: солнечно, +22°C ."

def handle_how_are_you(match):
    return "У меня всё отлично! Спасибо, что спросили."

def handle_time(match):
    current_time = datetime.now().strftime("%H:%M:%S")
    return f"Сейчас {current_time}"

def handle_addition(match):
    try:
        a = float(match.group(1))
        b = float(match.group(2))
        return f"Результат: {a} + {b} = {a + b}"
    except (ValueError, IndexError):
        return "Не удалось выполнить сложение. Пожалуйста, используйте формат: сколько будет число + число"

def handle_set_name(match):
    return bot_instance.set_name(match)


handler_map = {
    "greeting": handle_greeting,
    "farewell": handle_farewell,
    "weather": handle_weather,
    "how_are_you": handle_how_are_you,
    "time": handle_time,
    "addition": handle_addition,
    "set_name": handle_set_name,
}

def process_message(message: str):

    for pattern, handler_key in patterns:
        match = pattern.search(message)  
        if match:
      
            handler = handler_map.get(handler_key)
            if handler:
                return handler(match)
    

    cleaned_message = clean_message(message)
    response = bot_instance.process(cleaned_message)
    

    if response == "Не понимаю запрос.":
        return "Я не понимаю запрос. Попробуйте спросить что-то другое."
    
    return response