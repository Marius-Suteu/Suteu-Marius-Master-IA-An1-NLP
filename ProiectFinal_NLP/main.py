import pandas as pd
from preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model_ml import CustomNaiveBayes
from utils import evaluate_advanced

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader
from model_lstm import LSTMClassifier
import torch.nn as nn
import torch.optim as optim

# 1. LOAD DATA
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true], ignore_index=True)

# 2. PREPROCESS
df["clean_text"] = df["text"].astype(str).apply(preprocess)
df = df[df["clean_text"].str.len() > 0]

X = df["clean_text"].values
y = df["label"].values

# 3. SPLIT 70/15/15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

# 4. MODEL 1 – CUSTOM NB
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    max_df=0.85,
    min_df=10,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

configs = [0.5, 1.0, 2.0]
best_f1 = 0
best_nb = None

from sklearn.metrics import f1_score

for alpha in configs:
    nb = CustomNaiveBayes(alpha=alpha)
    nb.fit(X_train_vec, y_train)
    preds = nb.predict(X_val_vec)
    f1 = f1_score(y_val, preds, average="macro")
    print(f"NB alpha={alpha} | Macro-F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_nb = nb

nb = best_nb
y_pred_nb = nb.predict(X_test_vec)
y_prob_nb = X_test_vec @ nb.feature_log_prob[1]

print("\n===== Custom Naive Bayes Results =====")
evaluate_advanced(y_test, y_pred_nb, y_prob_nb)

# 5. MODEL 2 – LSTM + ATTENTION
from collections import Counter
from itertools import chain

all_tokens = [t.split() for t in X_train]
word_counts = Counter(chain(*all_tokens))
vocab = {w: i + 1 for i, (w, _) in enumerate(word_counts.most_common(10000))}

def text_to_seq(text, vocab, max_len=100):
    seq = [vocab.get(tok, 0) for tok in text.split()]
    return seq[:max_len] + [0] * (max_len - len(seq))

X_train_seq = [text_to_seq(t, vocab) for t in X_train]
X_val_seq   = [text_to_seq(t, vocab) for t in X_val]
X_test_seq  = [text_to_seq(t, vocab) for t in X_test]

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(NewsDataset(X_train_seq, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(NewsDataset(X_val_seq, y_val), batch_size=32)
test_loader  = DataLoader(NewsDataset(X_test_seq, y_test), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMClassifier(vocab_size=len(vocab) + 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 6. TRAIN + EARLY STOPPING
best_val_loss = float("inf")
patience = 2
counter = 0

for epoch in range(10):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()

    print(f"Epoch {epoch+1} | Train={train_loss:.4f} | Val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# 7. EVALUATE LSTM
model.eval()
all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())
        all_probs.extend(probs.cpu().numpy())

print("\n===== LSTM Results =====")
evaluate_advanced(all_labels, all_preds, all_probs)
