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
MODELO_INICIAL = PROJETO / "modelo"
SAIDA = PROJETO / "modelo_ajustado"

EPOCAS = 8
LOTE = 16
SEMENTE = 42


def avaliar(modelo, carregador, criterio):
    modelo.eval()

    perda_total = 0.0
    acertos = 0
    total = 0

    with torch.inference_mode():
        for imagens, rotulos in carregador:
            saidas = modelo(imagens)
            perda = criterio(saidas, rotulos)

            perda_total += perda.item() * len(rotulos)
            acertos += (
                saidas.argmax(1) == rotulos
            ).sum().item()
            total += len(rotulos)

    return perda_total / total, acertos / total


def main():
    random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.set_num_threads(4)

    if SAIDA.exists():
        print("A pasta modelo_ajustado já existe.")
        print("Renomeie essa pasta antes de iniciar outra experiência.")
        return

    with open(
        MODELO_INICIAL / "configuracao.json",
        encoding="utf-8"
    ) as arquivo:
        config = json.load(arquivo)

    tamanho = tuple(config["tamanho_imagem"])

    # Variações somente nas imagens de treinamento.
    # Sem recorte central, para não remover moedas das bordas.
    preparo_treino = transforms.Compose([
        transforms.Resize(tamanho),
        transforms.RandomRotation(
            degrees=15,
            fill=(128, 128, 128)
        ),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            config["media"],
            config["desvio"]
        )
    ])

    # Validação mantém o mesmo preparo da interface.
    preparo_validacao = transforms.Compose([
        transforms.Resize(tamanho),
        transforms.ToTensor(),
        transforms.Normalize(
            config["media"],
            config["desvio"]
        )
    ])

    treino = datasets.ImageFolder(
        DADOS / "treino",
        transform=preparo_treino
    )

    validacao = datasets.ImageFolder(
        DADOS / "validacao",
        transform=preparo_validacao
    )

    if (
        treino.classes != config["classes"]
        or validacao.classes != config["classes"]
    ):
        raise ValueError("As classes não correspondem ao modelo inicial.")

    carregador_treino = DataLoader(
        treino,
        batch_size=LOTE,
        shuffle=True,
        num_workers=0
    )

    carregador_validacao = DataLoader(
        validacao,
        batch_size=LOTE,
        shuffle=False,
        num_workers=0
    )

    modelo = models.mobilenet_v2(weights=None)

    entradas = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(
        entradas,
        len(config["classes"])
    )

    checkpoint = torch.load(
        MODELO_INICIAL / "mobilenet_moedas.pth",
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["classes"] != config["classes"]:
        raise ValueError("Configuração e modelo inicial incompatíveis.")

    modelo.load_state_dict(checkpoint["model_state_dict"])

    # Congela a rede e libera somente os últimos quatro blocos.
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    for parametro in modelo.features[-4:].parameters():
        parametro.requires_grad = True

    for parametro in modelo.classifier.parameters():
        parametro.requires_grad = True

    otimizador = torch.optim.Adam([
        {
            "params": modelo.features[-4:].parameters(),
            "lr": 0.00001
        },
        {
            "params": modelo.classifier.parameters(),
            "lr": 0.0001
        }
    ])

    criterio = nn.CrossEntropyLoss()

    print("Conferindo o modelo inicial na validação...")

    melhor_perda, melhor_acuracia = avaliar(
        modelo,
        carregador_validacao,
        criterio
    )

    print(f"Validação inicial: {melhor_acuracia:.2%}")

    SAIDA.mkdir()

    # Preserva o modelo inicial como referência.
    # Só o substitui se houver melhoria na validação.
    def salvar(epoca):
        torch.save(
            {
                "model_state_dict": modelo.state_dict(),
                "classes": config["classes"],
                "epoca": epoca,
                "acuracia_validacao": melhor_acuracia
            },
            SAIDA / "mobilenet_moedas.pth"
        )

    salvar(0)

    config["metodo"] = (
        "Ajuste dos últimos quatro blocos e do classificador; "
        "variações de rotação, brilho, contraste e saturação."
    )

    with open(
        SAIDA / "configuracao.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(config, arquivo, ensure_ascii=False, indent=2)

    inicio = time.perf_counter()
    melhor_epoca = 0

    with open(
        SAIDA / "historico.csv",
        "w",
        newline="",
        encoding="utf-8"
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
            # Preserva as estatísticas de normalização da rede.
            modelo.eval()
            modelo.classifier.train()

            perda_total = 0.0
            acertos = 0
            total = 0

            print(f"\nÉpoca {epoca}/{EPOCAS}")

            for numero, (imagens, rotulos) in enumerate(
                carregador_treino, 1
            ):
                otimizador.zero_grad()

                saidas = modelo(imagens)
                perda = criterio(saidas, rotulos)

                perda.backward()
                otimizador.step()

                perda_total += perda.item() * len(rotulos)
                acertos += (
                    saidas.argmax(1) == rotulos
                ).sum().item()
                total += len(rotulos)

                if numero % 20 == 0 or numero == len(carregador_treino):
                    print(
                        f"  Lote {numero}/{len(carregador_treino)}",
                        flush=True
                    )

            perda_treino = perda_total / total
            acuracia_treino = acertos / total

            perda_validacao, acuracia_validacao = avaliar(
                modelo,
                carregador_validacao,
                criterio
            )

            print(
                f"Treino: {acuracia_treino:.2%} | "
                f"Validação: {acuracia_validacao:.2%}"
            )

            escritor.writerow([
                epoca,
                perda_treino,
                acuracia_treino,
                perda_validacao,
                acuracia_validacao
            ])
            arquivo.flush()

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
                melhor_epoca = epoca
                salvar(epoca)

    minutos = (time.perf_counter() - inicio) / 60

    print("\nAjuste concluído!")
    print(f"Melhor validação: {melhor_acuracia:.2%}")
    print(f"Melhor época: {melhor_epoca}")
    print(f"Tempo do ajuste: {minutos:.1f} minutos")
    print(f"Arquivos salvos em: {SAIDA}")

    if melhor_epoca == 0:
        print("O modelo inicial foi mantido: não houve melhoria na validação.")

    print("A interface ainda usa o modelo original.")
    print("O conjunto de teste não foi usado neste ajuste.")


if __name__ == "__main__":
    main()