import re
import csv
import requests
import pandas as pd
from transformers import AutoTokenizer
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import datasets
from datasets import load_dataset
import re, random
from sklearn.model_selection import train_test_split

def load_cleen_dataset(url):

  # Создаём папку 'data' на уровень выше (../data)
  data_dir = os.path.join("..", "data")
  os.makedirs(data_dir, exist_ok=True)

  # Загрузка данных
  response = requests.get(url, verify=False)
  response.raise_for_status()
  data = response.text.splitlines()

  # Сохраняем сырой датасет
  output_path = os.path.join(data_dir, "raw_dataset.csv")
  pd.DataFrame({"tweet": data}).to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

  # Подготовка текста

  def clean_string(text):
      text = text.lower()
      text = re.sub(r"http\S+|www\S+", "", text)   #ссылки
      text = re.sub(r"@\w+", "", text)             #упоминания
      text = re.sub(r'[^a-z0-9\s]', '', text)
      text = re.sub(r'\s+', ' ', text).strip()
      return text

  cleaned = [clean_string(t) for t in data]

  #Сохранение результата
  output_path = os.path.join(data_dir, "dataset_processed.csv")
  df = pd.DataFrame({"tweets": cleaned})
  df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

  return df

def train_test_split_df(df, min_seq_len=4, test=False):
    
  # Создаём папку 'data' на уровень выше (../data)
  data_dir = os.path.join("..", "data")
  os.makedirs(data_dir, exist_ok=True)

  #Преобразование в Dataset
  dataset = datasets.Dataset.from_pandas(df)
  # Удаление None
  dataset = dataset.filter(lambda example: example['tweets'] is not None)

  if test:
    max_texts_count = 1000 #для теста
  else:
    max_texts_count = len(df)

  texts = [line for line in dataset["tweets"] if len(line.split()) > min_seq_len]

  train_val_texts, test_texts = train_test_split(texts[:max_texts_count], test_size=0.2, random_state=42)
  train_texts, val_texts = train_test_split(train_val_texts, test_size=0.1, random_state=42)

  train_df = pd.DataFrame(train_texts, columns=["tweets"])
  val_df = pd.DataFrame(val_texts, columns=["tweets"])
  test_df = pd.DataFrame(test_texts, columns=["tweets"])
  
  output_path = os.path.join(data_dir, "train.csv")

  #Сохранение в CSV файлы
  train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
  val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
  test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

  return train_texts, val_texts, test_texts