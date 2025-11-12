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
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import evaluate
from transformers import BertTokenizerFast

# Модель
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1, pad_token_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits