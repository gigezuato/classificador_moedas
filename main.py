import json
import torch
from torch import nn
from torchvision import models, transforms
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk, ImageOps


# Cores da interface
FUNDO = "#F3F5F9"
AZUL_ESCURO = "#172D46"
AZUL = "#245EDB"
CINZA = "#65758A"
BRANCO = "#FFFFFF"


class Aplicacao(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Identificação de moedas brasileiras")
        self.geometry("960x690")
        self.minsize(900, 650)
        self.configure(bg=FUNDO)

        self.imagem_original = None
        self.foto_previa = None

        self.option_add("*Font", ("Segoe UI", 11))

        self.criar_interface()

    def criar_interface(self):
        # Cabeçalho
        cabecalho = tk.Frame(
            self,
            bg=AZUL_ESCURO,
            padx=32,
            pady=22
        )
        cabecalho.pack(fill="x")

        tk.Label(
            cabecalho,
            text="Identificação de moedas brasileiras",
            font=("Segoe UI", 22, "bold"),
            bg=AZUL_ESCURO,
            fg=BRANCO
        ).pack(anchor="w")

        tk.Label(
            cabecalho,
            text="Uma imagem. Uma moeda.",
            bg=AZUL_ESCURO,
            fg="#C9D5E4"
        ).pack(anchor="w", pady=(7, 0))

        # Orientação
        tk.Label(
            self,
            text=(
                "Envie uma foto com uma única moeda brasileira de "
                "5, 10, 25, 50 centavos ou 1 real."
            ),
            bg=FUNDO,
            fg=CINZA,
            wraplength=850,
            justify="left"
        ).pack(anchor="w", padx=32, pady=(24, 16))

        # Área central
        conteudo = tk.Frame(self, bg=FUNDO)
        conteudo.pack(
            fill="both",
            expand=True,
            padx=32
        )

        conteudo.columnconfigure(0, weight=3)
        conteudo.columnconfigure(1, weight=2)
        conteudo.rowconfigure(0, weight=1)

        # Painel da imagem
        painel_imagem = tk.Frame(
            conteudo,
            bg=BRANCO,
            padx=22,
            pady=20
        )
        painel_imagem.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 18)
        )

        tk.Label(
            painel_imagem,
            text="1  Selecione sua imagem",
            font=("Segoe UI", 14, "bold"),
            bg=BRANCO,
            fg=AZUL_ESCURO
        ).pack(anchor="w")

        # O Frame mantém o espaço da prévia estável.
        area_previa = tk.Frame(
            painel_imagem,
            bg=FUNDO,
            width=420,
            height=280
        )
        area_previa.pack(
            fill="both",
            expand=True,
            pady=16
        )
        area_previa.pack_propagate(False)

        self.previa = tk.Label(
            area_previa,
            text="A prévia da imagem aparecerá aqui",
            bg=FUNDO,
            fg=CINZA
        )
        self.previa.pack(fill="both", expand=True)

        self.nome_arquivo = tk.Label(
            painel_imagem,
            text="Formatos aceitos: JPG e PNG",
            bg=BRANCO,
            fg=CINZA,
            wraplength=390
        )
        self.nome_arquivo.pack()

        tk.Button(
            painel_imagem,
            text="Selecionar imagem",
            command=self.selecionar_imagem,
            bg=AZUL,
            fg=BRANCO,
            activebackground="#194DB8",
            activeforeground=BRANCO,
            relief="flat",
            cursor="hand2",
            padx=24,
            pady=10
        ).pack(fill="x", pady=(14, 0))

        # Painel do resultado
        painel_resultado = tk.Frame(
            conteudo,
            bg=BRANCO,
            padx=22,
            pady=20
        )
        painel_resultado.grid(
            row=0,
            column=1,
            sticky="nsew"
        )

        tk.Label(
            painel_resultado,
            text="2  Resultado",
            font=("Segoe UI", 14, "bold"),
            bg=BRANCO,
            fg=AZUL_ESCURO
        ).pack(anchor="w")

        self.titulo_resultado = tk.Label(
            painel_resultado,
            text="Aguardando imagem",
            font=("Segoe UI", 17, "bold"),
            bg=BRANCO,
            fg=AZUL_ESCURO,
            wraplength=270
        )
        self.titulo_resultado.pack(pady=(45, 14))

        self.mensagem = tk.Label(
            painel_resultado,
            text="Selecione uma foto para começar.",
            bg=BRANCO,
            fg=CINZA,
            wraplength=260,
            justify="center"
        )
        self.mensagem.pack(pady=(0, 24))

        self.botao_identificar = tk.Button(
            painel_resultado,
            text="Identificar moeda",
            command=self.identificar_moeda,
            state=tk.DISABLED,
            bg=AZUL,
            fg=BRANCO,
            activebackground="#194DB8",
            activeforeground=BRANCO,
            disabledforeground="#8995A6",
            relief="flat",
            cursor="hand2",
            padx=16,
            pady=10
        )
        self.botao_identificar.pack(fill="x")

        tk.Button(
            painel_resultado,
            text="Limpar",
            command=self.limpar,
            bg=BRANCO,
            fg=CINZA,
            relief="flat",
            cursor="hand2",
            pady=8
        ).pack(pady=10)

        tk.Label(
            painel_resultado,
            text=(
                "Valores previstos no projeto\n\n"
                "R$ 0,05  •  R$ 0,10  •  R$ 0,25\n"
                "R$ 0,50  •  R$ 1,00"
            ),
            bg=BRANCO,
            fg=CINZA,
            justify="center"
        ).pack(side="bottom", pady=15)

        # Rodapé
        tk.Label(
            self,
            text=(
                "Protótipo acadêmico • "
                "Processamento de imagens e Machine Learning"
            ),
            font=("Segoe UI", 10),
            bg=FUNDO,
            fg=CINZA
        ).pack(pady=20)

    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione uma foto de uma moeda",
            filetypes=[
                ("Imagens JPG e PNG", "*.jpg *.jpeg *.png")
            ]
        )

        # Se cancelar, mantém a imagem anterior.
        if not caminho:
            return

        try:
            with Image.open(caminho) as arquivo:
                # Corrige a orientação de fotos de celular.
                imagem = ImageOps.exif_transpose(arquivo)
                imagem = imagem.convert("RGB")
                imagem.load()

            # Guarda a imagem original para a classificação futura.
            self.imagem_original = imagem

            # Redimensiona apenas a prévia, preservando a proporção.
            miniatura = ImageOps.contain(imagem, (390, 250))
            self.foto_previa = ImageTk.PhotoImage(miniatura)

            self.previa.config(
                image=self.foto_previa,
                text=""
            )

            self.nome_arquivo.config(
                text=Path(caminho).name
            )

            self.titulo_resultado.config(
                text="Imagem carregada"
            )

            self.mensagem.config(
                text="Clique em Identificar moeda para continuar."
            )

            self.botao_identificar.config(
                state=tk.NORMAL
            )

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError
        ):
            self.limpar()

            self.titulo_resultado.config(
                text="Imagem inválida"
            )

            self.mensagem.config(
                text=(
                    "Não foi possível abrir esse arquivo. "
                    "Selecione outra imagem."
                )
            )

    def identificar_moeda(self):
        if self.imagem_original is None:
            return

        self.botao_identificar.config(state=tk.DISABLED)
        self.titulo_resultado.config(text="Analisando...")
        self.mensagem.config(text="Aguarde um instante.")
        self.update_idletasks()

        try:
            # Carrega o modelo apenas na primeira classificação.
            if not hasattr(self, "modelo"):
                projeto = Path(__file__).resolve().parent
                pasta_modelo = projeto / "modelo"

                caminho_modelo = pasta_modelo / "mobilenet_moedas.pth"
                caminho_config = pasta_modelo / "configuracao.json"

                if not caminho_modelo.exists() or not caminho_config.exists():
                    raise FileNotFoundError(
                        "Coloque mobilenet_moedas.pth e configuracao.json "
                        "na pasta modelo, ao lado do main.py."
                    )

                with open(caminho_config, encoding="utf-8") as arquivo:
                    config = json.load(arquivo)

                checkpoint = torch.load(
                    caminho_modelo,
                    map_location="cpu",
                    weights_only=True
                )

                if checkpoint["classes"] != config["classes"]:
                    raise ValueError(
                        "O modelo e a configuração possuem classes diferentes."
                    )

                modelo = models.mobilenet_v2(weights=None)
                entradas = modelo.classifier[1].in_features

                modelo.classifier[1] = nn.Linear(
                    entradas,
                    len(config["classes"])
                )

                modelo.load_state_dict(
                    checkpoint["model_state_dict"]
                )
                modelo.eval()

                preparo = transforms.Compose([
                    transforms.Resize(
                        tuple(config["tamanho_imagem"])
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        config["media"],
                        config["desvio"]
                    )
                ])

                # Só guarda depois de carregar tudo corretamente.
                self.modelo = modelo
                self.preparo = preparo
                self.valores = config["valores"]

            # Usa a imagem original, com o mesmo preparo do treino.
            entrada = self.preparo(
                self.imagem_original.convert("RGB")
            ).unsqueeze(0)

            with torch.inference_mode():
                saida = self.modelo(entrada)
                indice = saida.argmax(dim=1).item()

            valor = self.valores[indice]

            self.titulo_resultado.config(text=valor)
            self.mensagem.config(
                text=(
                    "Valor previsto para a moeda da imagem.\n\n"
                    "O sistema não verifica se a foto contém uma moeda. "
                    "Imagens fora das condições indicadas podem gerar "
                    "resultados incorretos."
                )
            )
            
        except Exception as erro:
            self.titulo_resultado.config(
                text="Não foi possível classificar"
            )
            self.mensagem.config(
                text="Confira os arquivos do modelo e o erro no terminal."
            )
            print(f"\nErro na classificação: {erro}")

        finally:
            self.botao_identificar.config(state=tk.NORMAL)

    def limpar(self):
        self.imagem_original = None

        self.previa.config(
            image="",
            text="A prévia da imagem aparecerá aqui"
        )

        self.foto_previa = None

        self.nome_arquivo.config(
            text="Formatos aceitos: JPG e PNG"
        )

        self.titulo_resultado.config(
            text="Aguardando imagem"
        )

        self.mensagem.config(
            text="Selecione uma foto para começar."
        )

        self.botao_identificar.config(
            state=tk.DISABLED
        )


if __name__ == "__main__":
    app = Aplicacao()
    app.mainloop()