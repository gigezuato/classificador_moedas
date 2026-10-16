from pathlib import Path
import csv
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJETO = Path('C:/Users/Heitor/GitHub/classificador_moedas/all').resolve().parent
PASTA_MODELO = PROJETO / "modelo"
PASTA_TESTE = PROJETO / "dataset" / "teste"
RESULTADOS = PROJETO / "resultados"


def main():
    torch.set_num_threads(4)

    caminho_modelo = PASTA_MODELO / "mobilenet_moedas.pth"
    caminho_configuracao = PASTA_MODELO / "configuracao.json"

    for caminho in [
        caminho_modelo,
        caminho_configuracao,
        PASTA_TESTE
    ]:
        if not caminho.exists():
            print(f"Não encontrado: {caminho}")
            return

    with open(caminho_configuracao, encoding="utf-8") as arquivo:
        configuracao = json.load(arquivo)

    classes = configuracao["classes"]
    valores = configuracao["valores"]

    preparo = transforms.Compose([
        transforms.Resize(tuple(configuracao["tamanho_imagem"])),
        transforms.ToTensor(),
        transforms.Normalize(
            configuracao["media"],
            configuracao["desvio"]
        )
    ])

    base = datasets.ImageFolder(
        PASTA_TESTE,
        transform=preparo
    )

    if base.classes != classes:
        raise ValueError(
            "As classes do teste não correspondem às do treinamento."
        )

    carregador = DataLoader(
        base,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # Os pesos treinados serão carregados do arquivo local.
    modelo = models.mobilenet_v2(weights=None)
    entradas = modelo.classifier[1].in_features
    modelo.classifier[1] = nn.Linear(entradas, len(classes))

    checkpoint = torch.load(
        caminho_modelo,
        map_location="cpu",
        weights_only=True
    )

    if checkpoint["classes"] != classes:
        raise ValueError("A configuração não corresponde ao modelo salvo.")

    modelo.load_state_dict(checkpoint["model_state_dict"])
    modelo.eval()

    matriz = torch.zeros(
        (len(classes), len(classes)),
        dtype=torch.int64
    )

    registros = []
    perda_total = 0.0
    criterio = nn.CrossEntropyLoss(reduction="sum")
    posicao = 0
    inicio = time.perf_counter()

    print(f"Avaliando {len(base)} imagens de teste...")

    with torch.inference_mode():
        for numero, (imagens, rotulos) in enumerate(carregador, 1):
            saidas = modelo(imagens)
            previstas = saidas.argmax(dim=1)

            perda_total += criterio(saidas, rotulos).item()

            for real, prevista in zip(rotulos.tolist(), previstas.tolist()):
                matriz[real, prevista] += 1

                caminho_imagem = Path(base.samples[posicao][0])

                registros.append([
                    str(caminho_imagem.relative_to(PASTA_TESTE)),
                    valores[real],
                    valores[prevista],
                    real == prevista
                ])

                posicao += 1

            if numero % 10 == 0 or numero == len(carregador):
                print(f"  Lote {numero}/{len(carregador)}")

    segundos = time.perf_counter() - inicio

    total = int(matriz.sum().item())
    acertos = int(matriz.diag().sum().item())
    acuracia = acertos / total

    RESULTADOS.mkdir(exist_ok=True)

    with open(
        RESULTADOS / "previsoes_teste.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow([
            "imagem",
            "valor_real",
            "valor_previsto",
            "acertou"
        ])
        escritor.writerows(registros)

    with open(
        RESULTADOS / "matriz_confusao.csv",
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")
        escritor.writerow(["Real / Previsto"] + valores)

        for indice, linha in enumerate(matriz.tolist()):
            escritor.writerow([valores[indice]] + linha)

    metricas_classes = []
    linhas_relatorio = [
        "AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE",
        "",
        f"Total de imagens: {total}",
        f"Acertos: {acertos}",
        f"Erros: {total - acertos}",
        f"Acurácia: {acuracia:.2%}",
        f"Perda média: {perda_total / total:.4f}",
        f"Tempo de avaliação: {segundos:.2f} segundos",
        "",
        "RESULTADOS POR CLASSE",
        ""
    ]

    for indice, valor in enumerate(valores):
        verdadeiros_positivos = int(matriz[indice, indice].item())
        quantidade_real = int(matriz[indice].sum().item())
        quantidade_prevista = int(matriz[:, indice].sum().item())

        precisao = (
            verdadeiros_positivos / quantidade_prevista
            if quantidade_prevista else 0.0
        )
        recall = (
            verdadeiros_positivos / quantidade_real
            if quantidade_real else 0.0
        )
        f1 = (
            2 * precisao * recall / (precisao + recall)
            if precisao + recall else 0.0
        )

        metricas_classes.append({
            "valor": valor,
            "imagens": quantidade_real,
            "acertos": verdadeiros_positivos,
            "precisao": precisao,
            "recall": recall,
            "f1": f1
        })

        linhas_relatorio.append(
            f"{valor}: {verdadeiros_positivos}/{quantidade_real} acertos | "
            f"Precisão: {precisao:.2%} | "
            f"Recall: {recall:.2%} | F1: {f1:.2%}"
        )

    f1_macro = sum(item["f1"] for item in metricas_classes) / len(classes)
    linhas_relatorio.extend(["", f"F1 macro: {f1_macro:.2%}"])

    relatorio = "\n".join(linhas_relatorio)

    (RESULTADOS / "relatorio_teste.txt").write_text(
        relatorio,
        encoding="utf-8"
    )

    resumo = {
        "total": total,
        "acertos": acertos,
        "acuracia": acuracia,
        "perda_media": perda_total / total,
        "tempo_avaliacao_segundos": segundos,
        "f1_macro": f1_macro,
        "por_classe": metricas_classes
    }

    with open(
        RESULTADOS / "metricas_teste.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(resumo, arquivo, ensure_ascii=False, indent=2)

    print("\n" + relatorio)
    print(f"\nArquivos salvos em: {RESULTADOS}")


if __name__ == "__main__":
    main()