from torch.utils.data import Dataset
import torch

class NextWordDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=8):
        #Безопасная настройка pad_token (обязательно для GPT-2!)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_id = tokenizer.pad_token_id
        self.samples = []
        
        for line in texts:
            
            #Добавляем EOS если он есть
            if tokenizer.eos_token_id is not None:
                line += tokenizer.eos_token

            token_ids = tokenizer.encode(line, add_special_tokens=False)
            n = len(token_ids)
            
            if n < seq_len + 1: # нужно минимум seq_len + 1 токенов!
                continue

            for i in range(n - seq_len):
                x = token_ids[i : i + seq_len] # контекст
                y_token = token_ids[i + seq_len] #следующий токен
                # y: [pad, pad, ..., y_token] -длина seq_len, целевой токен в конце
                y = [self.pad_id] * (seq_len - 1) + [y_token]

                self.samples.append((x, y))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

