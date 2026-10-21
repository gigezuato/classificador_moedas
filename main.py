import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageOps

import torch
from torch import nn
from torchvision import models, transforms


PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = PROJETO / "modelo"
PREFERENCIAS = PROJETO / "preferencias_interface.json"

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Cada par representa: modo claro, modo escuro.
FUNDO = ("#EEF2F7", "#0C1522")
PAINEL = ("#FFFFFF", "#172538")
PREVIA = ("#E7EDF5", "#101C2D")
TEXTO = ("#172F48", "#F0F5FC")
SECUNDARIO = ("#475C73", "#BAC9DB")
BORDA = ("#CDD8E5", "#354860")
GUIA = ("#FFF3DB", "#342B1C")
DESTAQUE = ("#795019", "#F2CD8B")

AZUL = "#245BD2"
AZUL_HOVER = "#1948AE"

BOTAO_SECUNDARIO = ("#DCE6F2", "#2C415E")
BOTAO_SECUNDARIO_HOVER = ("#CAD9EC", "#3A5273")

DESATIVADO = ("#D5DFEC", "#34455E")
TEXTO_DESATIVADO = ("#44566C", "#C3CFDE")


class Aplicacao(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Identificação de moedas brasileiras")
        self.geometry("1100x820")
        self.minsize(960, 760)
        self.configure(fg_color=FUNDO)

        self.fonte = "Segoe UI"
        self.imagem_original = None
        self.imagem_previa = None
        self.modelo = None
        self.nome_atual = None
        self.previa_agendada = None

        self.modo_escuro = False
        try:
            with PREFERENCIAS.open(encoding="utf-8") as arquivo:
                preferencias = json.load(arquivo)
                self.modo_escuro = preferencias.get("tema") == "escuro"
        except (OSError, ValueError, AttributeError):
            pass

        ctk.set_appearance_mode(
            "dark" if self.modo_escuro else "light"
        )

        self.criar_interface()

    def fonte_texto(self, tamanho=16, negrito=False):
        return ctk.CTkFont(
            family=self.fonte,
            size=tamanho,
            weight="bold" if negrito else "normal",
        )

    def criar_botao(self, pai, texto, comando, principal=False):
        return ctk.CTkButton(
            pai,
            text=texto,
            command=comando,
            height=48,
            corner_radius=14,
            font=self.fonte_texto(17, True),
            fg_color=AZUL if principal else BOTAO_SECUNDARIO,
            hover_color=(
                AZUL_HOVER
                if principal
                else BOTAO_SECUNDARIO_HOVER
            ),
            text_color="#FFFFFF" if principal else TEXTO,
            text_color_disabled=TEXTO_DESATIVADO,
        )

    def criar_interface(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # Cabeçalho sem título grande.
        cabecalho = ctk.CTkFrame(
            self,
            fg_color=("#17324F", "#102137"),
            corner_radius=0,
        )
        cabecalho.grid(row=0, column=0, sticky="ew")
        cabecalho.grid_columnconfigure(1, weight=1)

        moeda = tk.Canvas(
            cabecalho,
            width=62,
            height=62,
            bg="#17324F" if not self.modo_escuro else "#102137",
            highlightthickness=0,
        )
        moeda.grid(row=0, column=0, padx=(25, 16), pady=18)
        moeda.create_oval(
            5, 5, 57, 57,
            fill="#EFC680",
            outline="#FFE0AA",
            width=2,
        )
        moeda.create_oval(
            11, 11, 51, 51,
            outline="#906122",
            width=2,
        )
        moeda.create_text(
            31, 31,
            text="R$",
            fill="#49321A",
            font=(self.fonte, 18, "bold"),
        )
        self.icone_moeda = moeda

        ctk.CTkLabel(
            cabecalho,
            text="Uma imagem. Uma moeda.",
            text_color="#FFFFFF",
            font=self.fonte_texto(22, True),
        ).grid(row=0, column=1, sticky="w", padx=(0, 12))

        self.botao_tema = ctk.CTkButton(
            cabecalho,
            text=(
                "Modo claro"
                if self.modo_escuro
                else "Modo escuro"
            ),
            command=self.alternar_tema,
            height=44,
            width=145,
            corner_radius=14,
            fg_color="#E4ECF7",
            hover_color="#C9D9ED",
            text_color="#15304C",
            font=self.fonte_texto(16, True),
        )
        self.botao_tema.grid(
            row=0, column=2, padx=(12, 25), pady=18
        )

        # Orientações.
        guia = ctk.CTkFrame(
            self,
            fg_color=GUIA,
            corner_radius=18,
        )
        guia.grid(
            row=1, column=0,
            sticky="ew",
            padx=24, pady=(18, 16),
        )
        guia.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            guia,
            text="Como preparar sua foto",
            text_color=DESTAQUE,
            font=self.fonte_texto(18, True),
        ).grid(
            row=0, column=0,
            sticky="w",
            padx=20, pady=(13, 3),
        )

        self.orientacao = ctk.CTkLabel(
            guia,
            text=(
                "Use fundo branco e liso, com boa iluminação. "
                "A moeda deve aparecer inteira, próxima e centralizada, "
                "ocupando boa parte da foto. Evite sombras fortes, "
                "reflexos e dedos ou objetos cobrindo a moeda."
            ),
            text_color=TEXTO,
            font=self.fonte_texto(16),
            justify="left",
            anchor="w",
            wraplength=970,
        )
        self.orientacao.grid(
            row=1, column=0,
            sticky="ew",
            padx=20, pady=(0, 14),
        )

        guia.bind(
            "<Configure>",
            lambda evento: self.orientacao.configure(
                wraplength=max(300, evento.width - 40)
            ),
        )

        # Painéis principais.
        conteudo = ctk.CTkFrame(self, fg_color="transparent")
        conteudo.grid(
            row=2, column=0,
            sticky="nsew",
            padx=24, pady=(0, 14),
        )
        conteudo.grid_columnconfigure(
            0, weight=3, uniform="paineis"
        )
        conteudo.grid_columnconfigure(
            1, weight=2, uniform="paineis"
        )
        conteudo.grid_rowconfigure(0, weight=1)

        esquerda = ctk.CTkFrame(
            conteudo,
            fg_color=PAINEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDA,
        )
        esquerda.grid(
            row=0, column=0,
            sticky="nsew",
            padx=(0, 16),
        )
        esquerda.grid_columnconfigure(0, weight=1)
        esquerda.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            esquerda,
            text="01  /  SUA IMAGEM",
            text_color=DESTAQUE,
            font=self.fonte_texto(15, True),
        ).grid(
            row=0, column=0,
            sticky="w",
            padx=22, pady=(20, 4),
        )

        ctk.CTkLabel(
            esquerda,
            text="Selecione sua moeda",
            text_color=TEXTO,
            font=self.fonte_texto(25, True),
        ).grid(
            row=1, column=0,
            sticky="w",
            padx=22, pady=(0, 14),
        )

        self.area_previa = ctk.CTkFrame(
            esquerda,
            fg_color=PREVIA,
            corner_radius=18,
            height=240,
        )
        self.area_previa.grid(
            row=2, column=0,
            sticky="nsew",
            padx=20,
        )
        self.area_previa.grid_propagate(False)
        self.area_previa.grid_columnconfigure(0, weight=1)
        self.area_previa.grid_rowconfigure(0, weight=1)

        self.previa = ctk.CTkLabel(
            self.area_previa,
            text="Sua imagem aparecerá aqui\n\nFormatos JPG e PNG",
            text_color=SECUNDARIO,
            font=self.fonte_texto(18),
            justify="center",
        )
        self.previa.grid(
            row=0, column=0,
            sticky="nsew",
            padx=12, pady=12,
        )

        self.area_previa.bind(
            "<Configure>", self.agendar_previa
        )

        self.nome_arquivo = ctk.CTkLabel(
            esquerda,
            text="Nenhuma imagem selecionada",
            text_color=SECUNDARIO,
            font=self.fonte_texto(14),
            wraplength=460,
        )
        self.nome_arquivo.grid(
            row=3, column=0,
            sticky="ew",
            padx=22, pady=10,
        )

        self.criar_botao(
            esquerda,
            "Selecionar imagem",
            self.selecionar_imagem,
            principal=True,
        ).grid(
            row=4, column=0,
            sticky="ew",
            padx=20, pady=(0, 20),
        )

        direita = ctk.CTkFrame(
            conteudo,
            fg_color=PAINEL,
            corner_radius=22,
            border_width=1,
            border_color=BORDA,
        )
        direita.grid(row=0, column=1, sticky="nsew")
        direita.grid_columnconfigure(0, weight=1)
        direita.grid_rowconfigure(7, weight=1)

        ctk.CTkLabel(
            direita,
            text="02  /  RESULTADO",
            text_color=DESTAQUE,
            font=self.fonte_texto(15, True),
        ).grid(
            row=0, column=0,
            sticky="w",
            padx=22, pady=(20, 16),
        )

        self.status = ctk.CTkLabel(
            direita,
            text="AGUARDANDO IMAGEM",
            text_color=SECUNDARIO,
            font=self.fonte_texto(14, True),
        )
        self.status.grid(
            row=1, column=0,
            padx=20, pady=(6, 8),
        )

        self.titulo_resultado = ctk.CTkLabel(
            direita,
            text="Qual é o valor?",
            text_color=TEXTO,
            font=self.fonte_texto(34, True),
            wraplength=310,
        )
        self.titulo_resultado.grid(
            row=2, column=0,
            padx=20, pady=(0, 12),
        )

        self.mensagem = ctk.CTkLabel(
            direita,
            text="Selecione uma foto para começar.",
            text_color=SECUNDARIO,
            font=self.fonte_texto(17),
            wraplength=300,
            justify="center",
        )
        self.mensagem.grid(
            row=3, column=0,
            padx=22, pady=(0, 20),
        )

        self.botao_identificar = self.criar_botao(
            direita,
            "Identificar moeda",
            self.identificar_moeda,
            principal=True,
        )
        self.botao_identificar.grid(
            row=4, column=0,
            sticky="ew",
            padx=22, pady=(0, 10),
        )
        self.habilitar_classificacao(False)

        self.criar_botao(
            direita,
            "Limpar imagem",
            self.limpar,
        ).grid(
            row=5, column=0,
            sticky="ew",
            padx=22,
        )

        valores = ctk.CTkFrame(
            direita,
            fg_color=PREVIA,
            corner_radius=16,
        )
        valores.grid(
            row=6, column=0,
            sticky="ew",
            padx=22, pady=(22, 18),
        )

        ctk.CTkLabel(
            valores,
            text="VALORES PREVISTOS",
            text_color=SECUNDARIO,
            font=self.fonte_texto(14, True),
        ).pack(pady=(12, 5))

        ctk.CTkLabel(
            valores,
            text="R$ 0,05  •  R$ 0,10  •  R$ 0,25\nR$ 0,50  •  R$ 1,00",
            text_color=TEXTO,
            font=self.fonte_texto(18, True),
            justify="center",
        ).pack(padx=10, pady=(0, 12))

        ctk.CTkLabel(
            self,
            text="Protótipo acadêmico  •  PDI e Machine Learning",
            text_color=SECUNDARIO,
            font=self.fonte_texto(14),
        ).grid(row=3, column=0, pady=(0, 14))

    def alternar_tema(self):
        self.modo_escuro = not self.modo_escuro
        tema = "escuro" if self.modo_escuro else "claro"

        ctk.set_appearance_mode(
            "dark" if self.modo_escuro else "light"
        )

        self.botao_tema.configure(
            text="Modo claro" if self.modo_escuro else "Modo escuro"
        )

        self.icone_moeda.configure(
            bg="#102137" if self.modo_escuro else "#17324F"
        )

        try:
            with PREFERENCIAS.open("w", encoding="utf-8") as arquivo:
                json.dump({"tema": tema}, arquivo)
        except OSError:
            pass

    def habilitar_classificacao(self, habilitar):
        self.botao_identificar.configure(
            state="normal" if habilitar else "disabled",
            fg_color=AZUL if habilitar else DESATIVADO,
        )

    def agendar_previa(self, evento=None):
        if self.previa_agendada is not None:
            self.after_cancel(self.previa_agendada)

        self.previa_agendada = self.after(
            100, self.atualizar_previa
        )

    def atualizar_previa(self):
        self.previa_agendada = None

        if self.imagem_original is None:
            return

        largura = max(1, self.area_previa.winfo_width() - 28)
        altura = max(1, self.area_previa.winfo_height() - 28)

        proporcao = min(
            largura / self.imagem_original.width,
            altura / self.imagem_original.height,
        )

        tamanho = (
            max(1, int(self.imagem_original.width * proporcao)),
            max(1, int(self.imagem_original.height * proporcao)),
        )

        self.imagem_previa = ctk.CTkImage(
            light_image=self.imagem_original,
            dark_image=self.imagem_original,
            size=tamanho,
        )

        self.previa.configure(
            image=self.imagem_previa,
            text="",
        )

    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione uma foto da moeda",
            filetypes=[
                ("Imagens JPG e PNG", "*.jpg *.jpeg *.png")
            ],
        )

        if not caminho:
            return

        try:
            with Image.open(caminho) as arquivo:
                imagem = ImageOps.exif_transpose(arquivo).convert("RGB")
                imagem.load()

            self.imagem_original = imagem
            self.nome_atual = Path(caminho).name

            self.nome_arquivo.configure(text=self.nome_atual)
            self.agendar_previa()

            self.status.configure(text="IMAGEM CARREGADA")
            self.titulo_resultado.configure(text="Tudo pronto!")
            self.mensagem.configure(
                text="Clique em Identificar moeda para analisar a foto."
            )
            self.habilitar_classificacao(True)

        except (
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            messagebox.showerror(
                "Imagem inválida",
                "Não foi possível abrir o arquivo. "
                "Selecione outra imagem JPG ou PNG.",
            )

    def carregar_modelo(self):
        if self.modelo is not None:
            return

        caminho_modelo = PASTA_MODELO / "mobilenet_moedas.pth"
        caminho_config = PASTA_MODELO / "configuracao.json"

        if not caminho_modelo.is_file() or not caminho_config.is_file():
            raise FileNotFoundError(
                "Não encontrei mobilenet_moedas.pth e configuracao.json "
                f"na pasta: {PASTA_MODELO}"
            )

        with caminho_config.open(encoding="utf-8") as arquivo:
            config = json.load(arquivo)

        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["classes"] != config["classes"]:
            raise ValueError(
                "O modelo e a configuração possuem classes diferentes."
            )

        modelo = models.mobilenet_v2(weights=None)
        entradas = modelo.classifier[1].in_features

        modelo.classifier[1] = nn.Linear(
            entradas,
            len(config["classes"]),
        )

        modelo.load_state_dict(checkpoint["model_state_dict"])
        modelo.eval()

        preparo = transforms.Compose([
            transforms.Resize(tuple(config["tamanho_imagem"])),
            transforms.ToTensor(),
            transforms.Normalize(
                config["media"],
                config["desvio"],
            ),
        ])

        self.modelo = modelo
        self.preparo = preparo
        self.valores = config["valores"]

        print(f"Modelo carregado: {caminho_modelo.resolve()}")

    def identificar_moeda(self):
        if self.imagem_original is None:
            return

        self.habilitar_classificacao(False)
        self.status.configure(text="ANALISANDO")
        self.titulo_resultado.configure(text="Um instante…")
        self.mensagem.configure(text="Consultando o modelo.")
        self.update_idletasks()

        try:
            self.carregar_modelo()

            entrada = self.preparo(
                self.imagem_original.convert("RGB")
            ).unsqueeze(0)

            with torch.inference_mode():
                saida = self.modelo(entrada)
                indice = saida.argmax(dim=1).item()

            valor = self.valores[indice]

            self.status.configure(text="VALOR PREVISTO")
            self.titulo_resultado.configure(text=valor)
            self.mensagem.configure(
                text=(
                    "A previsão pode conter erros. "
                    "O sistema não verifica se a foto contém uma moeda."
                )
            )

            print(f"Imagem: {self.nome_atual} | Previsão: {valor}")

        except Exception as erro:
            self.status.configure(text="ERRO NA ANÁLISE")
            self.titulo_resultado.configure(
                text="Não foi possível classificar"
            )
            self.mensagem.configure(
                text="Confira os arquivos do modelo "
                "e o erro no terminal."
            )
            print(f"\nErro na classificação: {erro}")

        finally:
            self.habilitar_classificacao(True)

    def limpar(self):
        self.imagem_original = None
        self.nome_atual = None

        self.previa.configure(
            image=None,
            text="Sua imagem aparecerá aqui\n\nFormatos JPG e PNG",
        )
        self.imagem_previa = None

        self.nome_arquivo.configure(
            text="Nenhuma imagem selecionada"
        )
        self.status.configure(text="AGUARDANDO IMAGEM")
        self.titulo_resultado.configure(text="Qual é o valor?")
        self.mensagem.configure(
            text="Selecione uma foto para começar."
        )
        self.habilitar_classificacao(False)


if __name__ == "__main__":
    app = Aplicacao()
    app.mainloop()