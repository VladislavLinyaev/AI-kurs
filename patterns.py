import re

patterns = [
    (re.compile(r"^(привет|здравствуй|добрый день)$", re.IGNORECASE), "greeting"),
    (re.compile(r"^(пока|до свидания)$", re.IGNORECASE), "farewell"),
    (re.compile(r"погода в ([а-яА-Яa-zA-Z\- ]+)", re.IGNORECASE), "weather"),
    (re.compile(r"как (у тебя )?дела", re.IGNORECASE), "how_are_you"),
    (re.compile(r"(сколько|какое) время", re.IGNORECASE), "time"),
    (re.compile(r"сколько будет (\d+)\s*\+\s*(\d+)", re.IGNORECASE), "addition"),
    (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), "set_name"),
]


class ChatBot:
    def __init__(self):
        self.name = None
        self.patterns = []
        self.register_patterns()
    
    def register_patterns(self):
        self.patterns.append(
            (re.compile(r"привет", re.IGNORECASE), self.greet)
        )
        self.patterns.append(
            (re.compile(r"меня зовут ([а-яА-Яa-zA-Z]+)", re.IGNORECASE), self.set_name)
        )
    
    def set_name(self, match):
        self.name = match.group(1)
        return f"Приятно познакомиться, {self.name}!"
    
    def greet(self, match):
        if self.name:
            return f"Здравствуйте, {self.name}!"
        return "Здравствуйте!"
    
    def process(self, message):

        message = message.lower().strip()
        
        for pattern, handler in self.patterns:
            match = pattern.search(message)
            if match:
                return handler(match)
        return "Не понимаю запрос."


bot_instance = ChatBot()