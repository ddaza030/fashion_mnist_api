from torch import nn


class NNFashionMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(500, 500),
            nn.ReLU(),
            self.dropout,
            nn.Linear(500, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
