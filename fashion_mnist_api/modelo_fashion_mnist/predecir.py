import io
import os
from PIL import Image
from torchvision import transforms
import torch

from modelo_fashion_mnist.modelo import NNFashionMnist


def predecir_fashion_mnist(
        imagen_bytes,
):
    imagen_pil = Image.open(io.BytesIO(imagen_bytes))

    transformador = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    imagen_tensor = transformador(imagen_pil).unsqueeze(0)

    modelo = NNFashionMnist()

    ruta_completa = os.path.join('input', 'model_nn_fashion_mnist.pth')
    pesos = torch.load(ruta_completa, map_location=torch.device('cpu'))
    modelo.load_state_dict(pesos)
    modelo.eval()

    clases = {
        0: 'Camiseta/Top',
        1: 'Pantalón',
        2: 'Suéter',
        3: 'Vestido',
        4: 'Abrigo',
        5: 'Sandalia',
        6: 'Camisa',
        7: 'Zapato',
        8: 'Bolsa',
        9: 'Bota'
    }

    with torch.no_grad():
        logits = modelo(imagen_tensor)
        predicciones = torch.argmax(logits, dim=1)
        clase_predicha = clases[predicciones.item()]

    return clase_predicha
