# train_bert.py
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  # Импортируем AdamW из torch.optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

print("=" * 60)
print("🤖 FINE-TUNING BERT НА ВАШЕМ DATASET.CSV")
print("=" * 60)

# 1. Загрузка датасета
print("\n📂 Загрузка dataset.csv...")
df = pd.read_csv("dataset.csv")
print(f"   ✅ Загружено {len(df)} примеров")
print(f"   📊 Интенты:\n{df['intent'].value_counts()}")

# 2. Маппинг - id2label должен иметь СТРОКОВЫЕ ключи
unique_intents = df['intent'].unique()
label2id = {label: idx for idx, label in enumerate(unique_intents)}
id2label = {str(idx): label for idx, label in enumerate(unique_intents)}  # КЛЮЧИ - СТРОКИ!
num_labels = len(unique_intents)

print(f"\n   📋 Количество интентов: {num_labels}")
print(f"   📋 label2id: {label2id}")
print(f"   📋 id2label: {id2label}")

# 3. Подготовка данных
df['label'] = df['intent'].map(label2id)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
print(f"\n✂️ Train: {len(train_texts)} примеров, Validation: {len(val_texts)} примеров")

# 4. Загрузка модели
MODEL_NAME = "DeepPavlov/rubert-base-cased"
print(f"\n📚 Загрузка модели {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"   ✅ Модель на устройстве: {device}")

# 5. Токенизация
print("\n🔄 Токенизация...")

def tokenize_texts(texts, labels):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    return encodings, torch.tensor(labels)

train_encodings, train_labels_tensor = tokenize_texts(train_texts, train_labels)
val_encodings, val_labels_tensor = tokenize_texts(val_texts, val_labels)

# 6. Dataset
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels_tensor)
val_dataset = IntentDataset(val_encodings, val_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# 7. Обучение
print("\n🧠 Обучение...")
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 6

for epoch in range(num_epochs):
    # Train
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"   Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# 8. Сохранение
print("\n💾 Сохранение модели...")
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")
print("   ✅ Модель сохранена в './intent_model'")

# 9. Проверка
print("\n🔍 Проверка сохраненной модели...")
print(f"   id2label: {model.config.id2label}")
print(f"   label2id: {model.config.label2id}")

# 10. Тест
print("\n🧪 Тест:")
test_phrases = ["привет", "какая погода", "сколько время", "пока", "2+2", "спасибо", "помощь"]
for phrase in test_phrases:
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        probs = torch.softmax(outputs.logits, dim=1)
        confidence = probs[0][pred].item()
        intent = id2label[str(pred)]
        print(f"   '{phrase}' -> {intent} (уверенность: {confidence:.2%})")

print("\n" + "=" * 60)
print("✅ FINE-TUNING BERT ЗАВЕРШЕН!")
print("=" * 60)