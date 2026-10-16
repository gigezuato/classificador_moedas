from pathlib import Path
import shutil

# Pasta onde este código está salvo
PROJETO = Path('C:/Users/Heitor/GitHub/classificador_moedas/all').resolve().parent

# Pasta com as fotos extraídas
ORIGEM = PROJETO / "all"

# Nova pasta organizada por valor
DESTINO = PROJETO / "dados_organizados"

CLASSES = {
    "5": "5_centavos",
    "10": "10_centavos",
    "25": "25_centavos",
    "50": "50_centavos",
    "100": "1_real"
}


def organizar():
    if not ORIGEM.is_dir():
        print(f"Pasta não encontrada: {ORIGEM}")
        print("Confira o nome da pasta com as fotos.")
        return

    contagem = {classe: 0 for classe in CLASSES.values()}
    ignoradas = 0

    # Procura fotos também nas subpastas, como moedas/all.
    arquivos = sorted(
        arquivo
        for arquivo in ORIGEM.rglob("*")
        if arquivo.is_file()
        and arquivo.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not arquivos:
        print("Nenhuma foto encontrada dentro da pasta moedas.")
        return

    for arquivo in arquivos:
        # Exemplo: 100_1477154436.jpg → 100
        valor = arquivo.stem.split("_")[0]

        if valor not in CLASSES:
            ignoradas += 1
            continue

        classe = CLASSES[valor]
        pasta_classe = DESTINO / classe
        pasta_classe.mkdir(parents=True, exist_ok=True)

        # Mantém a estrutura de origem para evitar sobrescrever
        # fotos com o mesmo nome em subpastas diferentes.
        relativo = arquivo.relative_to(ORIGEM)
        destino = pasta_classe / relativo
        destino.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(arquivo, destino)
        contagem[classe] += 1

    print("\nOrganização concluída!")
    print("As fotos originais foram preservadas.\n")

    for classe, quantidade in contagem.items():
        print(f"{classe}: {quantidade} fotos")

    print(f"\nFotos com nome não reconhecido: {ignoradas}")
    print(f"Destino: {DESTINO}")


if __name__ == "__main__":
    organizar()