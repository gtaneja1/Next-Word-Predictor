import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
import torch.optim as optim
import nltk
nltk.download('punkt_tab')
import fitz

doc = fitz.open("text.pdf")
text = ""

for page in doc:
    text += page.get_text()

tokens = word_tokenize(text.lower())
tokens = [word for word in tokens if word.isalpha()]

print(tokens[:20])
print(tokens)
vocab = sorted(set(tokens))
word2idx = {w: i + 1 for i, w in enumerate(vocab)}  # Use word2idx consistently
idx2word = {i: w for w, i in word2idx.items()}
print(word2idx)

print(word2idx.get("researchers", 0))
vocab_size = len(word2idx) + 1
sequence_length = 3
sequences = []  # initializing the slicing model process

for i in range(len(tokens) - sequence_length):
    seq = tokens[i:i + sequence_length]
    target = tokens[i + sequence_length]
    sequences.append((
        [word2idx.get(word, 0) for word in seq],   # get index or 0 if not found
        word2idx.get(target, 0)                    # same for target
    ))
    print(seq)
    print(target)
    print(word2idx[target])

# Dataset class
class WordDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = [torch.tensor(x[0]) for x in sequences]
        self.targets = [torch.tensor(x[1]) for x in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

dataset = WordDataset(sequences)
dataLoader = DataLoader(dataset, batch_size=2, shuffle=True)

# LSTM model class
class NextWordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)         # out: (batch_size, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])  # (batch_size, hidden_dim) → (batch_size, vocab_size)
        return out

loss_history = []

embed_dim = 64
hidden_dim = 128
epochs = 208

# initializing the model
model = NextWordLSTM(vocab_size, embed_dim, hidden_dim)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataLoader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Prediction function
def predict_next_word(model, seed_text, num_predict=1):
    model.eval()  # evaluation mode
    words = word_tokenize(seed_text.lower())
    words = [w for w in words if w.isalpha()]
    for _ in range(num_predict):
        x = [word2idx.get(w, 0) for w in words[-sequence_length:]]  # use last N words
        x = torch.tensor(x).unsqueeze(0)  # batch shape: (1, sequence_length)
        output = model(x)
        predicted_idx = output.argmax(dim=1).item()
        predicted_word = idx2word.get(predicted_idx, "<unk>")
        words.append(predicted_word)
    return " ".join(words)

print(predict_next_word(model, "applications to", num_predict=2))

# RNN model class
class NextWordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)            # (batch, seq_len, embed_dim)
        out, _ = self.rnn(x)         # out: (batch, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :]) # only last timestep's output
        return out


            # ...existing code...
   