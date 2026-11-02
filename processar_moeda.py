import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def processar_imagem(imagem, caminho_json):
    """Usa a anotação do Labelme para orientar a segmentação."""
    with Path(caminho_json).open(encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    rgb = np.array(imagem.convert("RGB"))
    altura, largura = rgb.shape[:2]

    if (
        dados["imageWidth"] != largura
        or dados["imageHeight"] != altura
    ):
        raise ValueError(
            "A anotação não tem as mesmas dimensões da imagem."
        )

    formas = [
        forma
        for forma in dados["shapes"]
        if forma["label"] == "moeda"
        and forma["shape_type"] == "polygon"
    ]

    if len(formas) != 1:
        raise ValueError(
            "A anotação precisa conter um polígono chamado moeda."
        )

    pontos = np.asarray(formas[0]["points"], dtype=float)

    if len(pontos) < 3:
        raise ValueError("A marcação precisa de pelo menos três pontos.")

    x, y = np.floor(pontos.min(axis=0) - 5).astype(int)
    direita, baixo = np.ceil(pontos.max(axis=0) + 5).astype(int)

    x, y = max(0, x), max(0, y)
    direita = min(largura, direita)
    baixo = min(altura, baixo)

    if direita <= x or baixo <= y:
        raise ValueError("A marcação está fora da imagem.")

    if x == 0 and y == 0 and direita == largura and baixo == altura:
        raise ValueError("A marcação precisa deixar algum fundo de fora.")

    mascara = np.zeros((altura, largura), dtype=np.uint8)

    cv2.setRNGSeed(42)
    cv2.grabCut(
        rgb,
        mascara,
        (int(x), int(y), int(direita - x), int(baixo - y)),
        np.zeros((1, 65), dtype=np.float64),
        np.zeros((1, 65), dtype=np.float64),
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    mascara = np.where(
        (mascara == cv2.GC_FGD)
        | (mascara == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contornos:
        raise ValueError("Não foi possível separar a moeda.")

    mascara[:] = 0
    cv2.drawContours(
        mascara,
        [max(contornos, key=cv2.contourArea)],
        -1,
        255,
        -1,
    )

    resultado = np.full_like(rgb, 255)
    resultado[mascara > 0] = rgb[mascara > 0]

    return Image.fromarray(mascara), Image.fromarray(resultado)