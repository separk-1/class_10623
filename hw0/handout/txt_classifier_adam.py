# From: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import time
import torchvision.transforms as T
import re
import wandb

# Hyperparameters
EPOCHS = 5  # epoch
LR = 0.001  # learning rate
BATCH_SIZE = 8  # batch size for training
EMBED_DIM = 64 # embedding size in model
MAX_LEN = 1024 # maximum text input length

# Get cpu, gpu device for training.
# mps does not (yet) support nn.EmbeddingBag.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        text = self.data_frame.loc[idx, "article"]
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            text = self.transform(text)

        return text, label

class SimpleTokenizer:
    def __call__(self, text):
        # Add a space between punctuation and words
        text = re.sub(r'([.,:;!?()])', r' \1 ', text)
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize by splitting on whitespace
        return text.split()

class Vocab:
    def __init__(self, oov_token, pad_token):
        self.idx2str = []
        self.str2idx = {}
        self.oov_index = 0
        self.add_tokens([oov_token, pad_token])
        self.oov_idx = self[oov_token]
        self.pad_idx = self[pad_token]

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.str2idx:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)

    def __len__(self):
        return len(self.str2idx)

    def __getitem__(self, token):
        return self.str2idx.get(token, self.oov_index)

class CorpusInfo():
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.oov_token = '<OOV>' # out-of-vocabulary token
        self.pad_token = '<PAD>' # padding token
        
        self.vocab = Vocab(self.oov_token, self.pad_token)
        for text, _ in dataset:
            self.vocab.add_tokens(tokenizer(text))
        
        self.oov_idx = self.vocab[self.oov_token]
        self.pad_idx = self.vocab[self.pad_token]
        
        self.vocab_size = len(self.vocab)
        self.num_labels = len(set([label for (text, label) in dataset]))

class TextTransform():
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def tokenize_and_numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.vocab[token] for token in tokens]

    def __call__(self, text):
        return self.tokenize_and_numericalize(text)
    
class MaxLen():
    def __init__(self, max_len):
        self.max_len = max_len
        
    def __call__(self, x):
        if len(x) > self.max_len:
            x = x[:self.max_len]
        return x
    
class PadSequence():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        def to_int_tensor(x):
            return torch.from_numpy(np.array(x, dtype=np.int64))
        # Convert each sequence of tokens to a Tensor
        sequences = [to_int_tensor(x[0]) for x in batch]
        # Convert the full sequence of labels to a Tensor
        labels = to_int_tensor([x[1] for x in batch])
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.pad_idx)
        return sequences_padded, labels


def log_histogram_of_article_lengths(dataset):
    wandb.init(project="text-classification", name="article_length_histogram")

    # 기사 길이 측정 및 MAX_LEN 적용
    lengths = [min(len(str(text).split()), MAX_LEN) for text, _ in dataset if isinstance(text, str)]

    if not lengths:
        print("No valid data found for logging.")
        wandb.finish()
        return

    # 0부터 50 단위로 구간 나누기
    bin_edges = np.arange(0, MAX_LEN + 50, 50)
    hist, bin_edges = np.histogram(lengths, bins=bin_edges)

    # X축 범위를 문자열로 변환 (예: "0-50", "51-100")
    bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]

    # wandb 테이블로 x축과 y축 로깅
    wandb.log({"article_length_histogram": wandb.Table(columns=["Range", "Count"], data=list(zip(bin_labels, hist)))})

    print(f"Logged {len(lengths)} article lengths with 50-unit bins to wandb.")
    print(wandb.run.summary)

    wandb.finish()


def get_data():    
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=None,
    )
    tokenizer = SimpleTokenizer()
    corpus_info = CorpusInfo(train_data, tokenizer)
    transform_txt = T.Compose([
        TextTransform(corpus_info.tokenizer, corpus_info.vocab),
        MaxLen(MAX_LEN),
    ])
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=transform_txt,
    )
    val_data = CsvTextDataset(
        csv_file='./data/txt_val.csv',
        transform=transform_txt,
    )
    test_data = CsvTextDataset(
        csv_file='./data/txt_test.csv',
        transform=transform_txt,
    )

    collate_batch = PadSequence(corpus_info.pad_idx)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_batch)

    for X, y in train_dataloader:
        print(f"Shape of X [B, N]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return corpus_info, train_dataloader, val_dataloader, test_dataloader

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        text = text.to(self.embedding.weight.device)
        embedded = self.embedding(text)
        return self.fc(embedded)


def train_one_epoch(dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 5
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def main():
    corpus_info, train_dataloader, val_dataloader, test_dataloader = get_data()
    # 훈련 데이터셋 로드 (전처리 없이 원본 기사 길이 확인)
    train_dataset = CsvTextDataset(csv_file='./data/txt_train.csv', transform=None)

    # wandb를 통해 히스토그램 기록
    log_histogram_of_article_lengths(train_dataset)

    model = TextClassificationModel(corpus_info.vocab_size, EMBED_DIM, corpus_info.num_labels).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    total_accu = None    
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch)
        accu_val = evaluate(val_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model, criterion)
    print("test accuracy {:8.3f}".format(accu_test))

    
    print("Checking the results of val dataset.")
    accu_test = evaluate(val_dataloader, model, criterion)
    print("val accuracy {:8.3f}".format(accu_val))

if __name__ == '__main__':
    main()