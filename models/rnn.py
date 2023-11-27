import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,  # (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        # x: (batch, seq_len, embedding_dim)
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden_dim)
        out = self.fc(out[:, -1, :])
        # out: (batch, num_classes)
        return out



class RNN_modi(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(RNN_modi, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,  # (batch, seq, feature)
        )
        self.fc1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc2 = nn.Linear(2 * hidden_dim, num_classes)

        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc1.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        # x: (batch, seq_len, embedding_dim)
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden_dim)
        out = self.fc1(out[:, -1, :])
        # out: (batch, 2 * hidden_dim)
        out = F.relu(out)
        # out: (batch, 2 * hidden_dim)
        out = self.fc2(out)
        # out: (batch, num_classes)
        return out