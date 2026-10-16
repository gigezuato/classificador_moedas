from pathlib import Path
import csv
import json
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJETO = Path('C:/Users/Heitor/GitHub/classificador_moedas/all').resolve().parent
DADOS = PROJETO / "dataset"
SAIDA = PROJETO / "modelo"

EPOCAS = 10
TAMANHO_LOTE = 16
SEMENTE = 42

MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]

VALORES = {
    "5_centavos": "R$ 0,05",
    "10_centavos": "R$ 0,10",
    "25_centavos": "R$ 0,25",
    "50_centavos": "R$ 0,50",
    "1_real": "R$ 1,00"
}


def main():
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)

    # Limita o uso de threads para evitar sobrecarga.
    torch.set_num_threads(4)

    for nome in ["treino", "validacao"]:
        if not (DADOS / nome).is_dir():
            print(f"Pasta não encontrada: {DADOS / nome}")
            print("Execute separar_dados.py primeiro.")
            return

    SAIDA.mkdir(exist_ok=True)

    # Mantém a imagem inteira: não recorta o centro,
    # pois algumas moedas estão próximas das bordas.
    preparo = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO)
    ])

    bases = {
        nome: datasets.ImageFolder(
            DADOS / nome,
            transform=preparo
        )
        for nome in ["treino", "validacao"]
    }

    if bases["treino"].class_to_idx != bases["validacao"].class_to_idx:
        raise ValueError("As classes de treino e validação são diferentes.")

    classes = bases["treino"].classes

    if set(classes) != set(VALORES):
        raise ValueError(
            f"Classes encontradas: {classes}. "
            "Confira os nomes das cinco pastas."
        )

    print("Classes e índices:", bases["treino"].class_to_idx)
    print("Imagens de treino:", len(bases["treino"]))
    print("Imagens de validação:", len(bases["validacao"]))
    print("\nCarregando a MobileNetV2...")
    print("Na primeira execução, os pesos serão baixados.")

    modelo = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT
    )

    # Nesta primeira experiência, treinamos apenas
    # o classificador final e preservamos a base da rede.
    for parametro in modelo.features.parameters():
        parametro.requires_grad = False

    entradas = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entradas, len(classes))

    modelo.eval()

    # Extrai as características uma única vez.
    # Isso reduz o trabalho repetido no processador.
    caracteristicas = {}

    for nome, base in bases.items():
        carregador = DataLoader(
            base,
            batch_size=TAMANHO_LOTE,
            shuffle=False,
            num_workers=0
        )

        vetores = []
        rotulos = []

        print(f"\nPreparando características de {nome}...")

        with torch.no_grad():
            for lote, (imagens, etiquetas) in enumerate(carregador, 1):
                saida = modelo.features(imagens)
                saida = saida.mean(dim=(2, 3))

                vetores.append(saida)
                rotulos.append(etiquetas)

                if lote % 20 == 0 or lote == len(carregador):
                    print(f"  Lote {lote}/{len(carregador)}")

        caracteristicas[nome] = (
            torch.cat(vetores),
            torch.cat(rotulos)
        )

    treino_x, treino_y = caracteristicas["treino"]
    validacao_x, validacao_y = caracteristicas["validacao"]

    otimizador = torch.optim.Adam(
        modelo.classifier.parameters(),
        lr=0.001
    )
    criterio = nn.CrossEntropyLoss()

    melhor_acuracia = -1.0
    melhor_perda = float("inf")
    inicio = time.time()

    caminho_modelo = SAIDA / "mobilenet_moedas.pth"
    caminho_historico = SAIDA / "historico.csv"

    # Guarda a ordem real das classes e o preparo da entrada.
    configuracao = {
        "classes": classes,
        "valores": [VALORES[classe] for classe in classes],
        "tamanho_imagem": [224, 224],
        "media": MEDIA,
        "desvio": DESVIO,
        "preparo": "RGB e redimensionamento integral para 224x224",
        "metodo": "Base congelada e treinamento do classificador",
        "semente": SEMENTE
    }

    with open(SAIDA / "configuracao.json", "w", encoding="utf-8") as arquivo:
        json.dump(configuracao, arquivo, ensure_ascii=False, indent=2)

    with open(
        caminho_historico, "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow([
            "epoca",
            "perda_treino",
            "acuracia_treino",
            "perda_validacao",
            "acuracia_validacao"
        ])

        for epoca in range(1, EPOCAS + 1):
            modelo.classifier.train()

            ordem = torch.randperm(len(treino_y))
            perda_total = 0.0
            acertos = 0

            for inicio_lote in range(0, len(ordem), TAMANHO_LOTE):
                indices = ordem[
                    inicio_lote:inicio_lote + TAMANHO_LOTE
                ]

                x = treino_x[indices]
                y = treino_y[indices]

                otimizador.zero_grad()
                previsoes = modelo.classifier(x)
                perda = criterio(previsoes, y)

                perda.backward()
                otimizador.step()

                perda_total += perda.item() * len(y)
                acertos += (previsoes.argmax(1) == y).sum().item()

            perda_treino = perda_total / len(treino_y)
            acuracia_treino = acertos / len(treino_y)

            modelo.classifier.eval()

            with torch.no_grad():
                previsoes = modelo.classifier(validacao_x)
                perda_validacao = criterio(
                    previsoes, validacao_y
                ).item()
                acuracia_validacao = (
                    previsoes.argmax(1) == validacao_y
                ).float().mean().item()

            escritor.writerow([
                epoca,
                perda_treino,
                acuracia_treino,
                perda_validacao,
                acuracia_validacao
            ])
            arquivo.flush()

            print(
                f"Época {epoca}/{EPOCAS} | "
                f"Treino: {acuracia_treino:.1%} | "
                f"Validação: {acuracia_validacao:.1%}"
            )

            # Em empate de acurácia, escolhe a menor perda.
            melhorou = (
                acuracia_validacao > melhor_acuracia
                or (
                    acuracia_validacao == melhor_acuracia
                    and perda_validacao < melhor_perda
                )
            )

            if melhorou:
                melhor_acuracia = acuracia_validacao
                melhor_perda = perda_validacao

                torch.save(
                    {
                        "model_state_dict": modelo.state_dict(),
                        "classes": classes,
                        "epoca": epoca,
                        "acuracia_validacao": melhor_acuracia
                    },
                    caminho_modelo
                )

    minutos = (time.time() - inicio) / 60

    print("\nTreinamento concluído!")
    print(f"Melhor acurácia de validação: {melhor_acuracia:.1%}")
    print(f"Tempo da etapa de treinamento: {minutos:.1f} minutos")
    print(f"Modelo salvo em: {caminho_modelo}")
    print("O conjunto de teste ainda não foi utilizado.")


if __name__ == "__main__":
    main()