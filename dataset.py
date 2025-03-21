import json

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_seq_len
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.data = []
        with open(data_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip())['text'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]

        encode = self.tokenizer(
            text,
            max_length = self.max_seq_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        input_ids = encode['input_ids'].squeeze()
        loss_mask = input_ids != self.tokenizer.pad_token_id

        x = input_ids[:-1]
        y = input_ids[1:]
        loss_mask = loss_mask[1:]
        return x, y, loss_mask


if __name__ == '__main__':

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('./mini-deepseek-tokenizer')

    train_dataset = PretrainDataset(
        data_path = 'E:/pretrain_hq.jsonl',
        tokenizer = tokenizer,
        max_seq_len = 512
    )

    train_dataset[0]