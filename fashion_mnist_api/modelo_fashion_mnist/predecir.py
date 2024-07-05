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

    modelo = NNFashionMnist(capas_lineares=2, cantidad_neuronas_por_capa=500)

    ruta_completa = os.path.join('input', 'nombre.pth')
    pesos = torch.load(ruta_completa, map_location=torch.device('cpu'))
    modelo.load_state_dict(pesos)
    modelo.eval()

    with torch.no_grad():
        logits = modelo(imagen_tensor)
        predicciones = torch.argmax(logits, dim=1)
        clase_predicha = predicciones.item()

    return clase_predicha
