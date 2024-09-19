import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        # c_n: (num_layers * num_directions, batch, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        # 取出最后一个时间步的输出，并连接正向和反向隐藏状态
        out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        # out = F.relu(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
