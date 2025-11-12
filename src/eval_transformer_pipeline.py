import os
import sys
import torch
from tqdm import tqdm

# Добавляем src в путь (на случай запуска извне)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорты из соседних файлов
from data_utils import load_cleen_dataset, train_test_split_df
from eval_lstm import print_rouge_scores, print_rouge_scores_test
from transformers import pipeline
from transformers import AutoTokenizer

import warnings
import logging

#Игнорировать все Python-предупреждения
warnings.filterwarnings("ignore")

#Отключить INFO/WARNING от evaluate, transformers, urllib3
logging.getLogger("evaluate").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

print("Скачивание и подготовка данных...")

url = "https://code.s3.yandex.net/deep-learning/tweets.txt"

df = load_cleen_dataset(url) #загрузка и очистка

train_texts, val_texts, test_texts = train_test_split_df(df, min_seq_len=4, test=False) #разбиение на тест-валидацию-тренировочную

# Путь к файлу с весами модели
output_path = "../model/nextword_lstm_.pth"

# Определяем путь к скачанному токенизатору и модели-трансформер
script_dir = os.getcwd() #путь к нотбуку
model_dir = os.path.normpath(os.path.join(script_dir, "..", "model", "distilgpt2_local")) #к скачанному токенизатору и модели-трансформер

tokenizer = AutoTokenizer.from_pretrained(model_dir)


generator = pipeline(
    "text-generation",
    model=model_dir,
    tokenizer=model_dir,
    device=0 if torch.cuda.is_available() else -1
)

# Подготавливаем списки для ROUGE
preds_texts = []
targets_texts = []

# Генерация 1/4 на 10_000 примеров
for text in tqdm(test_texts[:10000], desc=f"Transformer|Generation [test]"):
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Берём первые 3/4 как seed
    split_idx = max(4, (3 * len(tokens)) // 4)
    seed_tokens = tokens[:split_idx]
    target_tokens = tokens[split_idx:]

    seed_text = tokenizer.decode(seed_tokens, skip_special_tokens=True)
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True).strip()

    # Генерируем
    result = generator(
        seed_text,
        max_new_tokens=len(target_tokens) + 5,  # + небольшой запас
        do_sample=True,
        top_k=50,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    gen_text = result[0]["generated_text"]

    # Очистка: один пробел, strip, убираем начальные пунктуационные артефакты
    pred_text = ' '.join(gen_text.split()).strip()
    
    # Обрезаем до длины target
    pred_words = pred_text.split()[:len(target_text.split())]
    pred_text = ' '.join(pred_words)

    if pred_text and target_text:
        preds_texts.append(pred_text)
        targets_texts.append(target_text)

#Вычисляем ROUGE
print_rouge_scores_test(preds_texts, targets_texts) #Вызов ф-ции для подсчета ROUGE на тесте