
from intent_classifier import intent_classifier


print("Проверка модели...")

test_phrases = [
    "привет",
    "погода в москве",
    "сколько время",
    "как дела",
    "пока"
]

for phrase in test_phrases:
    intent, conf = intent_classifier.predict_intent(phrase)
    print(f"'{phrase}' -> {intent} (уверенность: {conf:.2%})")