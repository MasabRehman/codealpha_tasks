import torch
import torch.nn as nn
import torch.nn.functional as F

class HandwrittenCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(HandwrittenCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.25)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((2, 2))
        self.drop3 = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.drop_fc = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop2(self.pool2(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop3(self.pool3(F.relu(self.bn4(self.conv4(x)))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        logits = self.fc2(x)
        return logits


class CRNN_CTC(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super(CRNN_CTC, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.2), # <--- Added CNN Dropout
            
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Dropout2d(0.2), # <--- Added CNN Dropout
            
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((4, 1), (4, 1)),
        )
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.5 # <--- Massive LSTM Dropout to cure overfitting!
        )
        
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)
        
    def forward(self, x):
        conv = self.cnn(x)
        seq = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(seq)
        return self.fc(rnn_out)
