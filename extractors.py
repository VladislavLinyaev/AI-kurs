
import re
from datetime import datetime
import spacy


nlp = spacy.load("ru_core_news_sm")


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
            print(f"[DEBUG] NER: {ent.text!r} -> {lemma!r}")
            return lemma

    m = re.search(r"\bв[о]?\s+([А-ЯЁ][а-яёА-ЯЁ\-]+)", text)
    if m:
        word = m.group(1)
        word_doc = nlp(word)
        lemma = word_doc[0].lemma_.capitalize() if word_doc else word
        print(f"[DEBUG] Regex: {word!r} -> {lemma!r}")
        return lemma

    print(f"[DEBUG] Город не найден в: {text!r}")
    return None


def is_weather_query(text: str) -> bool:
    return "погода" in text.lower()