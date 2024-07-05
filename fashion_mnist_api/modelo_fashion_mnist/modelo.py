from torch import nn


class NNFashionMnist(nn.Module):
    def __init__(self, capas_lineares, cantidad_neuronas_por_capa):
        super().__init__()

        layers = [nn.Flatten()]

        for i in range(capas_lineares):
            if i == 0:
                layers.append(nn.Linear(28 * 28, cantidad_neuronas_por_capa))
            else:
                layers.append(nn.Linear(cantidad_neuronas_por_capa, cantidad_neuronas_por_capa))
            layers.append(nn.ReLU())

        # Capa lineal final
        layers.append(nn.Linear(cantidad_neuronas_por_capa, 10))
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
