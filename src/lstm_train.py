import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Добавляем src в путь (на случай запуска извне)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты из соседних файлов
from data_utils import load_cleen_dataset, train_test_split_df
from next_token_dataset import NextWordDataset
from lstm_model import NextWordLSTM
from eval_lstm import print_rouge_scores


from transformers import BertTokenizerFast

print("Подготовка данных...")

url = "https://code.s3.yandex.net/deep-learning/tweets.txt"

df = load_cleen_dataset(url) # загрузка и очистка

train_texts, val_texts, test_texts = train_test_split_df(df, min_seq_len=4, test=False) #разбиение на тест-валидацию-тренировочную

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


seq_len = 8

train_dataset = NextWordDataset(train_texts, tokenizer, seq_len)
val_dataset = NextWordDataset(val_texts, tokenizer, seq_len)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)


#Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
pad_id = tokenizer.pad_token_id
model = NextWordLSTM(vocab_size, pad_token_id=pad_id).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def decode_tokens(tokens):
    return tokenizer.decode([t for t in tokens if t != pad_id], skip_special_tokens=True)

#генерация НЕСКОЛЬКИХ токенов
def generate_sequence(model, tokenizer, seed_tokens, max_new_tokens=10, seq_len=4):
    """
    Генерирует max_new_tokens новых токенов, начиная с seed_tokens.
    Использует скользящее окно длины seq_len.
    """
    model.eval()
    tokens = seed_tokens.copy()  # list of ints

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_seq = tokens[-seq_len:]
            x = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # (1, L)
            logits = model(x)  # (1, L, V)
            next_token_logits = logits[0, -1, :]  # (V,)
            next_token = torch.argmax(next_token_logits).item()
            tokens.append(next_token)
    return tokens

# Обучение + ВАЛИДАЦИЯ С ГЕНЕРАЦИЕЙ 1/4 ПРЕДЛОЖЕНИЯ
for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Train loss: {total_loss/len(train_loader):.4f}")

    #ВАЛИДАЦИЯ: генерация 1/4 ПРЕДЛОЖЕНИЯ
    model.eval()
    val_loss = 0
    preds_texts, targets_texts = [], []

    # Считаем лосс
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item()

    # генерация на полных предложениях из val_texts
    for text in val_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < seq_len + 2:  # нужно минимум что-то сгенерировать
            continue

        # Определяем seed: первые 3/4 токенов, но не меньше seq_len
        n = len(token_ids)
        seed_len = max(seq_len, (3 * n) // 4)
        gen_len = n - seed_len  # сколько токенов сгенерировать (1/4)

        if gen_len <= 0:
            continue

        seed_tokens = token_ids[:seed_len]
        target_tokens = token_ids[seed_len:]  # истинное продолжение

        # Генерируем ровно gen_len токенов
        generated_tokens = generate_sequence(
            model, tokenizer, seed_tokens, max_new_tokens=gen_len, seq_len=seq_len
        )
        generated_continuation = generated_tokens[seed_len:]  # только сгенерированная часть

        # Декодируем
        pred_text = decode_tokens(generated_continuation)
        target_text = decode_tokens(target_tokens)

        # Фильтруем пустые строки
        if pred_text.strip() and target_text.strip():
            preds_texts.append(pred_text)
            targets_texts.append(target_text)

    # Вычисляем ROUGE
    print_rouge_scores(val_loss, val_loader, preds_texts, targets_texts)


# Создаём папку 'models' на уровень выше (../models)
data_dir = os.path.join("..", "models")
os.makedirs(data_dir, exist_ok=True)
# Сохранение 
output_path = os.path.join(data_dir, "nextword_lstm.pth")

torch.save(model.state_dict(), output_path)