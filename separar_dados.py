from pathlib import Path
import hashlib
import random
import shutil

PROJETO = Path('C:/Users/Heitor/GitHub/classificador_moedas/all').resolve().parent
ORIGEM = PROJETO / "dados_organizados"
DESTINO = PROJETO / "dataset"

CLASSES = [
    "5_centavos",
    "10_centavos",
    "25_centavos",
    "50_centavos",
    "1_real"
]

SEMENTE = 42


def separar():
    if not ORIGEM.is_dir():
        print("Pasta dados_organizados não encontrada.")
        return

    # Evita misturar uma divisão nova com arquivos antigos.
    if DESTINO.exists():
        print("A pasta dataset já existe.")
        print("Não foi alterada. Use a divisão que já foi criada.")
        return

    grupos = {}
    hashes_classes = {}

    # Primeiro confere todas as classes.
    for classe in CLASSES:
        pasta = ORIGEM / classe

        arquivos = sorted(
            arquivo
            for arquivo in pasta.rglob("*")
            if arquivo.is_file()
            and arquivo.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not arquivos:
            print(f"Nenhuma imagem encontrada em: {pasta}")
            return

        grupos[classe] = {}

        for arquivo in arquivos:
            resumo = hashlib.sha256(arquivo.read_bytes()).hexdigest()

            # Uma foto idêntica não deve ter rótulos diferentes.
            if resumo in hashes_classes:
                if hashes_classes[resumo] != classe:
                    print("Imagem idêntica encontrada em classes diferentes:")
                    print(arquivo)
                    print("Confira os rótulos antes de continuar.")
                    return

            hashes_classes[resumo] = classe
            grupos[classe].setdefault(resumo, []).append(arquivo)

        if len(grupos[classe]) < 10:
            print(f"Poucas imagens diferentes na classe {classe}.")
            return

    sorteio = random.Random(SEMENTE)

    for classe in CLASSES:
        # Arquivos idênticos permanecem no mesmo conjunto.
        lotes = list(grupos[classe].values())
        sorteio.shuffle(lotes)

        limite_treino = int(len(lotes) * 0.70)
        limite_validacao = limite_treino + int(len(lotes) * 0.15)

        conjuntos = {
            "treino": lotes[:limite_treino],
            "validacao": lotes[limite_treino:limite_validacao],
            "teste": lotes[limite_validacao:]
        }

        print(f"\n{classe}")

        for conjunto, grupos_arquivos in conjuntos.items():
            pasta_destino = DESTINO / conjunto / classe
            pasta_destino.mkdir(parents=True, exist_ok=True)

            quantidade = 0

            for grupo in grupos_arquivos:
                for arquivo in grupo:
                    relativo = arquivo.relative_to(ORIGEM / classe)
                    destino = pasta_destino / relativo
                    destino.parent.mkdir(parents=True, exist_ok=True)

                    shutil.copy2(arquivo, destino)
                    quantidade += 1

            print(f"  {conjunto}: {quantidade} fotos")

    print("\nSeparação concluída!")
    print(f"Arquivos salvos em: {DESTINO}")
    print("As imagens originais foram preservadas.")


if __name__ == "__main__":
    separar()