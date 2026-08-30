"""
Task 2: Deep Learning Architectures for Speech Emotion Recognition
Includes 1D-CNN (Temporal MFCC) and Bidirectional LSTM models in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Speech1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Temporal MFCC sequences.
    Input Shape: (batch_size, n_mfcc, seq_len), e.g. (B, 40, 150)
    """
    def __init__(self, in_channels: int = 40, num_classes: int = 6):
        super(Speech1DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.2)
        
        # Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)
        
        # Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (B, 40, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.global_pool(x) # (B, 256, 1)
        x = x.squeeze(-1)       # (B, 256)
        
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        logits = self.fc2(x)
        return logits

class SpeechLSTM(nn.Module):
    """
    Bidirectional LSTM for sequential acoustic temporal dynamics.
    Input Shape: (batch_size, seq_len, in_features), e.g. (B, 150, 40)
    """
    def __init__(self, in_features: int = 40, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 6):
        super(SpeechLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.25 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (B, 40, T) -> Transpose to (B, T, 40)
        if x.shape[1] == 40 and x.shape[2] != 40:
            x = x.transpose(1, 2)
            
        lstm_out, _ = self.lstm(x) # (B, T, hidden_size * 2)
        # Average pooling across time
        pooled = torch.mean(lstm_out, dim=1) # (B, hidden_size * 2)
        
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

class SpeechCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM for Speech Emotion Recognition.
    """
    def __init__(self, in_channels: int = 40, num_classes: int = 7, hidden_size: int = 64):
        super(SpeechCNNLSTM, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        self.drop2 = nn.Dropout(0.3)
        
        # LSTM Sequence Modeler
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Classification Head
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # x shape: (B, 40, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))
        
        # Transpose for LSTM: (B, T', 128)
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x) # (B, T', hidden_size * 2)
        pooled = torch.mean(lstm_out, dim=1) # (B, hidden_size * 2)
        
        logits = self.fc(pooled)
        return logits


class SpeechRNN(nn.Module):
    """
    Standard RNN for sequential acoustic temporal dynamics.
    Input Shape: (batch_size, seq_len, in_features), e.g. (B, 150, 40)
    """
    def __init__(self, in_features: int = 40, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 7):
        super(SpeechRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='relu',
            dropout=0.25 if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        if x.shape[1] == 40 and x.shape[2] != 40:
            x = x.transpose(1, 2)
            
        rnn_out, _ = self.rnn(x) # (B, T, hidden_size)
        pooled = torch.mean(rnn_out, dim=1) # (B, hidden_size)
        
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

