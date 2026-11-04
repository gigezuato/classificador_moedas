import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageOps

import torch
from torch import nn
from torchvision import models, transforms


# ============================================================
# CAMINHOS E CONFIGURAÇÕES
# ============================================================

PROJETO = Path(__file__).resolve().parent
PASTA_MODELO = PROJETO / "modelo_ajustado"
PASTA_PDI = PROJETO / "resultados_pdi"

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# Paleta fixa: tema claro
FUNDO = "#F3F6FB"
PAINEL = "#FFFFFF"
PREVIA = "#EDF2F8"
TEXTO = "#172B4D"
SECUNDARIO = "#64748B"
BORDA = "#DCE4EF"
GUIA = "#EAF2FF"
DESTAQUE = "#2457A6"

COR_CABECALHO = "#142B4A"
COR_NAV = "#1E3A63"
COR_NAV_HOVER = "#2C4D7E"
TEXTO_NAV = "#C9D9F2"

TEXTO_PREVIA_VAZIA = "Clique aqui para escolher uma imagem\n\nFormatos JPG e PNG"

# Estados visuais do resultado: (fundo do selo, texto do selo, cor do título)
ESTADOS = {
    "neutro": ("#DDE6F2", "#64748B", "#174A91"),
    "pronto": ("#DBEAFE", "#1D4ED8", "#174A91"),
    "analisando": ("#FEF3C7", "#92400E", "#174A91"),
    "sucesso": ("#DCFCE7", "#166534", "#166534"),
    "erro": ("#FEE2E2", "#B91C1C", "#B91C1C"),
}

AZUL = "#2563EB"
AZUL_HOVER = "#1D4ED8"
BOTAO_SECUNDARIO = "#E8EEF7"
BOTAO_SECUNDARIO_HOVER = "#D8E3F2"
DESATIVADO = "#DCE5F0"
TEXTO_DESATIVADO = "#64748B"


class Aplicacao(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Identificação de moedas brasileiras")

        # Ajusta o tamanho inicial para telas menores (ex.: notebooks 1366x768)
        try:
            escala = float(self._get_window_scaling())
        except (AttributeError, TypeError, ValueError):
            escala = 1.0
        largura = min(1180, int(self.winfo_screenwidth() / escala) - 60)
        altura = min(880, int(self.winfo_screenheight() / escala) - 110)
        self.geometry(f"{largura}x{altura}")
        self.minsize(min(940, largura), min(680, altura))
        self.configure(fg_color=FUNDO)

        self.fonte = "Segoe UI"

        # Estado da classificação
        self.imagem_original = None
        self.imagem_previa = None
        self.modelo = None
        self.preparo = None
        self.valores = None
        self.nome_atual = None
        self.previa_agendada = None

        # Mantém referências das imagens exibidas na galeria PDI.
        # Isso evita que o Tkinter descarte as imagens.
        self.imagens_pdi = []

        self.criar_interface()
        self.after(150, self.carregar_galeria_pdi)

    def fonte_texto(self, tamanho=16, negrito=False):
        return ctk.CTkFont(
            family=self.fonte,
            size=tamanho,
            weight="bold" if negrito else "normal",
        )

    def criar_botao(self, pai, texto, comando, principal=False, altura=50, largura=140):
        return ctk.CTkButton(
            pai,
            text=texto,
            command=comando,
            width=largura,
            height=altura,
            corner_radius=13,
            font=self.fonte_texto(16, True),
            fg_color=AZUL if principal else BOTAO_SECUNDARIO,
            hover_color=AZUL_HOVER if principal else BOTAO_SECUNDARIO_HOVER,
            text_color="#FFFFFF" if principal else TEXTO,
            text_color_disabled=TEXTO_DESATIVADO,
        )

    def ajustar_quebra(self, rotulo, referencia, margem):
        """Faz o texto do rótulo quebrar de acordo com a largura do contêiner."""
        def atualizar(evento):
            try:
                largura = int(referencia._reverse_widget_scaling(evento.width))
            except AttributeError:
                largura = int(evento.width)
            largura = max(180, largura - margem)

            if getattr(rotulo, "_largura_quebra", None) == largura:
                return
            rotulo._largura_quebra = largura
            try:
                rotulo.configure(wraplength=largura)
            except (ValueError, tk.TclError):
                pass

        referencia.bind("<Configure>", atualizar)

    def atualizar_status(self, texto, estado="neutro"):
        fundo, cor_texto, cor_titulo = ESTADOS[estado]
        self.status.configure(text=texto, fg_color=fundo, text_color=cor_texto)
        self.titulo_resultado.configure(text_color=cor_titulo)

    def mostrar_pagina(self, chave):
        paginas = {
            "classificar": self.aba_classificador,
            "pdi": self.aba_pdi,
        }
        for nome, pagina in paginas.items():
            if nome == chave:
                pagina.grid(row=0, column=0, sticky="nsew")
            else:
                pagina.grid_remove()

        for nome, botao in self.botoes_nav.items():
            ativo = nome == chave
            botao.configure(
                fg_color=AZUL if ativo else COR_NAV,
                hover_color=AZUL_HOVER if ativo else COR_NAV_HOVER,
                text_color="#FFFFFF" if ativo else TEXTO_NAV,
            )

    # ========================================================
    # ESTRUTURA PRINCIPAL
    # ========================================================

    def criar_interface(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        cabecalho = ctk.CTkFrame(
            self,
            fg_color=COR_CABECALHO,
            corner_radius=0,
        )
        cabecalho.grid(row=0, column=0, sticky="ew")
        cabecalho.grid_columnconfigure(1, weight=1)

        icone = tk.Canvas(
            cabecalho,
            width=62,
            height=62,
            bg=COR_CABECALHO,
            highlightthickness=0,
        )
        icone.grid(row=0, column=0, padx=(25, 16), pady=10)
        icone.create_oval(
            5, 5, 57, 57,
            fill="#EFC680",
            outline="#FFE0AA",
            width=2,
        )
        icone.create_oval(
            11, 11, 51, 51,
            outline="#906122",
            width=2,
        )
        icone.create_text(
            31, 31,
            text="R$",
            fill="#49321A",
            font=(self.fonte, 18, "bold"),
        )

        titulos = ctk.CTkFrame(cabecalho, fg_color="transparent")
        titulos.grid(row=0, column=1, sticky="w", padx=(0, 20))

        ctk.CTkLabel(
            titulos,
            text="Identificação de moedas brasileiras",
            text_color="#FFFFFF",
            font=self.fonte_texto(21, True),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            titulos,
            text="PDI  •  MACHINE LEARNING",
            text_color=TEXTO_NAV,
            font=self.fonte_texto(12, True),
        ).grid(row=1, column=0, sticky="w")

        # Navegação principal
        nav = ctk.CTkFrame(cabecalho, fg_color="transparent", corner_radius=0)
        nav.grid(row=0, column=2, padx=(10, 25))

        self.botoes_nav = {}
        itens_nav = (
            ("classificar", "Classificar moeda"),
            ("pdi", "Demonstração de PDI"),
        )
        for coluna, (chave, texto) in enumerate(itens_nav):
            botao = ctk.CTkButton(
                nav,
                text=texto,
                width=160,
                height=40,
                corner_radius=20,
                font=self.fonte_texto(14, True),
                fg_color=COR_NAV,
                hover_color=COR_NAV_HOVER,
                text_color=TEXTO_NAV,
                command=lambda c=chave: self.mostrar_pagina(c),
            )
            botao.grid(row=0, column=coluna, padx=4)
            self.botoes_nav[chave] = botao

        # Linha de destaque abaixo do cabeçalho
        ctk.CTkFrame(
            cabecalho,
            height=3,
            fg_color="#3B82F6",
            corner_radius=0,
        ).grid(row=1, column=0, columnspan=3, sticky="ew")

        self.paginas = ctk.CTkFrame(self, fg_color="transparent")
        self.paginas.grid(
            row=1, column=0,
            sticky="nsew",
            padx=26, pady=(18, 8),
        )
        self.paginas.grid_columnconfigure(0, weight=1)
        self.paginas.grid_rowconfigure(0, weight=1)

        self.aba_classificador = ctk.CTkFrame(self.paginas, fg_color="transparent")
        self.aba_pdi = ctk.CTkFrame(self.paginas, fg_color="transparent")

        self.criar_aba_classificador()
        self.criar_aba_pdi()
        self.mostrar_pagina("classificar")

        ctk.CTkLabel(
            self,
            text="Protótipo acadêmico desenvolvido para fins educacionais",
            text_color=SECUNDARIO,
            font=self.fonte_texto(13),
        ).grid(row=2, column=0, pady=(0, 10))

    # ========================================================
    # ABA 1 - CLASSIFICAÇÃO
    # ========================================================

    def criar_aba_classificador(self):
        aba = self.aba_classificador
        aba.grid_columnconfigure(0, weight=1)
        aba.grid_rowconfigure(1, weight=1)

        guia = ctk.CTkFrame(
            aba,
            fg_color="#FFF6D6",
            corner_radius=18,
            border_width=2,
            border_color="#F2C94C",
        )
        guia.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 12))
        guia.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            guia,
            text="Orientações para a fotografia",
            text_color="#8A5A00",
            font=self.fonte_texto(16, True),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(10, 2))

        orientacao = (
            "Utilize uma foto com boa iluminação e a moeda inteira visível. "
            "Prefira um fundo simples e evite sombras fortes, reflexos ou "
            "objetos cobrindo a moeda. O sistema recebe uma imagem por vez."
        )
        texto_orientacao = ctk.CTkLabel(
            guia,
            text=orientacao,
            text_color="#5C4300",
            font=self.fonte_texto(14),
            justify="left",
            anchor="w",
            wraplength=900,
        )
        texto_orientacao.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))
        self.ajustar_quebra(texto_orientacao, guia, 40)

        conteudo = ctk.CTkFrame(aba, fg_color="transparent")
        conteudo.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
        conteudo.grid_columnconfigure(0, weight=3, uniform="paineis")
        conteudo.grid_columnconfigure(1, weight=2, uniform="paineis")
        conteudo.grid_rowconfigure(0, weight=1)

        # ----------------------------------------------------
        # Painel esquerdo: envio da imagem
        # ----------------------------------------------------
        esquerda = ctk.CTkFrame(
            conteudo,
            fg_color=PAINEL,
            corner_radius=20,
            border_width=1,
            border_color=BORDA,
        )
        esquerda.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        esquerda.grid_columnconfigure(0, weight=1)
        esquerda.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            esquerda,
            text="ETAPA 01  •  ENVIO DA IMAGEM",
            text_color=DESTAQUE,
            font=self.fonte_texto(14, True),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 2))

        ctk.CTkLabel(
            esquerda,
            text="Selecione uma moeda",
            text_color=TEXTO,
            font=self.fonte_texto(23, True),
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))

        self.area_previa = ctk.CTkFrame(
            esquerda,
            fg_color=PREVIA,
            corner_radius=18,
            border_width=2,
            border_color="#C9D8EC",
            height=240,
        )
        self.area_previa.grid(row=2, column=0, sticky="nsew", padx=18)
        self.area_previa.grid_propagate(False)
        self.area_previa.grid_columnconfigure(0, weight=1)
        self.area_previa.grid_rowconfigure(0, weight=1)

        self.previa = ctk.CTkLabel(
            self.area_previa,
            text=TEXTO_PREVIA_VAZIA,
            text_color=SECUNDARIO,
            font=self.fonte_texto(17),
            justify="center",
        )
        self.previa.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.area_previa.bind("<Configure>", self.agendar_previa)

        # Clicar na área vazia também abre o seletor de arquivos
        self.area_previa.bind("<Button-1>", self.clicar_previa)
        self.previa.bind("<Button-1>", self.clicar_previa)

        self.nome_arquivo = ctk.CTkLabel(
            esquerda,
            text="Nenhuma imagem selecionada",
            text_color=SECUNDARIO,
            font=self.fonte_texto(13),
            wraplength=500,
        )
        self.nome_arquivo.grid(row=3, column=0, sticky="ew", padx=20, pady=8)
        self.ajustar_quebra(self.nome_arquivo, esquerda, 40)

        botoes_esquerda = ctk.CTkFrame(esquerda, fg_color="transparent")
        botoes_esquerda.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 18))
        botoes_esquerda.grid_columnconfigure((0, 1), weight=1)

        self.criar_botao(
            botoes_esquerda,
            "Selecionar imagem",
            self.selecionar_imagem,
            principal=True,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.criar_botao(
            botoes_esquerda,
            "Limpar",
            self.limpar,
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # ----------------------------------------------------
        # Painel direito: resultado
        # ----------------------------------------------------
        direita = ctk.CTkFrame(
            conteudo,
            fg_color=PAINEL,
            corner_radius=20,
            border_width=1,
            border_color=BORDA,
        )
        direita.grid(row=0, column=1, sticky="nsew")
        direita.grid_columnconfigure(0, weight=1)
        direita.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            direita,
            text="ETAPA 02  •  RESULTADO",
            text_color=DESTAQUE,
            font=self.fonte_texto(14, True),
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(16, 10))

        caixa_resultado = ctk.CTkFrame(
            direita,
            fg_color=PREVIA,
            corner_radius=18,
        )
        caixa_resultado.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 12))
        caixa_resultado.grid_columnconfigure(0, weight=1)
        caixa_resultado.grid_rowconfigure((0, 4), weight=1)

        self.status = ctk.CTkLabel(
            caixa_resultado,
            text="AGUARDANDO IMAGEM",
            text_color=SECUNDARIO,
            font=self.fonte_texto(12, True),
            corner_radius=14,
            height=28,
        )
        self.status.grid(row=1, column=0, padx=16, pady=(14, 10))

        self.titulo_resultado = ctk.CTkLabel(
            caixa_resultado,
            text="Qual é o valor?",
            text_color="#174A91",
            font=self.fonte_texto(32, True),
            wraplength=280,
        )
        self.titulo_resultado.grid(row=2, column=0, padx=14, pady=(0, 10))
        self.ajustar_quebra(self.titulo_resultado, caixa_resultado, 32)

        self.mensagem = ctk.CTkLabel(
            caixa_resultado,
            text="Selecione uma foto para começar.",
            text_color=SECUNDARIO,
            font=self.fonte_texto(15),
            wraplength=270,
            justify="center",
        )
        self.mensagem.grid(row=3, column=0, padx=16, pady=(0, 14))
        self.ajustar_quebra(self.mensagem, caixa_resultado, 36)

        self.atualizar_status("AGUARDANDO IMAGEM", "neutro")

        self.botao_identificar = self.criar_botao(
            direita,
            "Identificar moeda",
            self.identificar_moeda,
            principal=True,
        )
        self.botao_identificar.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.habilitar_classificacao(False)

        valores = ctk.CTkFrame(
            direita,
            fg_color="#EDF3FC",
            corner_radius=16,
            border_width=1,
            border_color="#DCE7F5",
        )
        valores.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 10))
        valores.grid_columnconfigure((0, 1, 2, 3, 4), weight=1, uniform="valor")

        ctk.CTkLabel(
            valores,
            text="VALORES QUE O MODELO IDENTIFICA (R$)",
            text_color=SECUNDARIO,
            font=self.fonte_texto(12, True),
        ).grid(row=0, column=0, columnspan=5, pady=(12, 8))

        lista_valores = ("0,05", "0,10", "0,25", "0,50", "1,00")
        for coluna, valor in enumerate(lista_valores):
            esquerda_pad = 10 if coluna == 0 else 3
            direita_pad = 10 if coluna == len(lista_valores) - 1 else 3
            ctk.CTkLabel(
                valores,
                text=valor,
                text_color=TEXTO,
                fg_color=PAINEL,
                corner_radius=8,
                height=30,
                font=self.fonte_texto(13, True),
            ).grid(
                row=1,
                column=coluna,
                sticky="ew",
                padx=(esquerda_pad, direita_pad),
                pady=(0, 12),
            )

        aviso = ctk.CTkLabel(
            direita,
            text=(
                "A previsão é uma estimativa do modelo. "
                "O sistema não detecta automaticamente se a imagem contém uma moeda."
            ),
            text_color=SECUNDARIO,
            font=self.fonte_texto(12),
            wraplength=290,
            justify="center",
        )
        aviso.grid(row=4, column=0, padx=18, pady=(0, 16))
        self.ajustar_quebra(aviso, direita, 40)

    def clicar_previa(self, evento=None):
        if self.imagem_original is None:
            self.selecionar_imagem()

    def habilitar_classificacao(self, habilitar):
        if hasattr(self, "botao_identificar"):
            self.botao_identificar.configure(
                state="normal" if habilitar else "disabled",
                fg_color=AZUL if habilitar else DESATIVADO,
            )

    def agendar_previa(self, evento=None):
        if self.previa_agendada is not None:
            try:
                self.after_cancel(self.previa_agendada)
            except tk.TclError:
                pass

        self.previa_agendada = self.after(100, self.atualizar_previa)

    def atualizar_previa(self):
        self.previa_agendada = None
        if self.imagem_original is None:
            return

        try:
            area_largura = int(self.area_previa._reverse_widget_scaling(self.area_previa.winfo_width()))
            area_altura = int(self.area_previa._reverse_widget_scaling(self.area_previa.winfo_height()))
        except AttributeError:
            area_largura = self.area_previa.winfo_width()
            area_altura = self.area_previa.winfo_height()

        largura = max(1, area_largura - 28)
        altura = max(1, area_altura - 28)

        proporcao = min(
            largura / self.imagem_original.width,
            altura / self.imagem_original.height,
        )
        tamanho = (
            max(1, int(self.imagem_original.width * proporcao)),
            max(1, int(self.imagem_original.height * proporcao)),
        )

        nova_previa = ctk.CTkImage(
            light_image=self.imagem_original,
            size=tamanho,
        )
        self.previa.configure(image=nova_previa, text="")
        self.imagem_previa = nova_previa

    def selecionar_imagem(self):
        caminho = filedialog.askopenfilename(
            title="Selecione uma foto da moeda",
            filetypes=[("Imagens JPG e PNG", "*.jpg *.jpeg *.png")],
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

            self.atualizar_status("IMAGEM CARREGADA", "pronto")
            self.titulo_resultado.configure(text="Tudo pronto!")
            self.mensagem.configure(
                text="Clique em Identificar moeda para analisar a foto."
            )
            self.habilitar_classificacao(True)

        except (OSError, ValueError, Image.DecompressionBombError) as erro:
            messagebox.showerror(
                "Imagem inválida",
                f"Não foi possível abrir essa imagem.\n\nDetalhes: {erro}",
            )

    def carregar_modelo(self):
        if self.modelo is not None:
            return

        caminho_modelo = PASTA_MODELO / "mobilenet_moedas.pth"
        caminho_config = PASTA_MODELO / "configuracao.json"

        if not caminho_modelo.is_file() or not caminho_config.is_file():
            raise FileNotFoundError(
                "Não encontrei os arquivos do modelo ajustado.\n"
                f"Verifique a pasta:\n{PASTA_MODELO}"
            )

        with caminho_config.open(encoding="utf-8") as arquivo:
            config = json.load(arquivo)

        campos = ["classes", "valores", "tamanho_imagem", "media", "desvio"]
        faltantes = [campo for campo in campos if campo not in config]
        if faltantes:
            raise ValueError(
                "O arquivo configuracao.json não possui: "
                + ", ".join(faltantes)
            )

        checkpoint = torch.load(
            caminho_modelo,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint.get("classes") != config["classes"]:
            raise ValueError(
                "As classes do modelo e da configuração são diferentes."
            )

        if len(config["classes"]) != len(config["valores"]):
            raise ValueError(
                "A quantidade de classes e valores no arquivo de configuração não corresponde."
            )

        modelo = models.mobilenet_v2(weights=None)
        entradas = modelo.classifier[1].in_features
        modelo.classifier[1] = nn.Linear(entradas, len(config["classes"]))

        modelo.load_state_dict(checkpoint["model_state_dict"])
        modelo.eval()

        preparo = transforms.Compose([
            transforms.Resize(tuple(config["tamanho_imagem"])),
            transforms.ToTensor(),
            transforms.Normalize(config["media"], config["desvio"]),
        ])

        self.modelo = modelo
        self.preparo = preparo
        self.valores = config["valores"]

    def identificar_moeda(self):
        if self.imagem_original is None:
            return

        self.habilitar_classificacao(False)
        self.atualizar_status("ANALISANDO...", "analisando")
        self.titulo_resultado.configure(text="Um instante...")
        self.mensagem.configure(text="O modelo está analisando a imagem.")
        self.update_idletasks()

        try:
            self.carregar_modelo()

            inicio = time.perf_counter()
            entrada = self.preparo(
                self.imagem_original.convert("RGB")
            ).unsqueeze(0)

            with torch.inference_mode():
                saida = self.modelo(entrada)
                indice = int(saida.argmax(dim=1).item())

            if indice < 0 or indice >= len(self.valores):
                raise ValueError("O modelo retornou uma classe inválida.")

            valor = self.valores[indice]
            tempo = time.perf_counter() - inicio

            self.atualizar_status("VALOR PREVISTO", "sucesso")
            self.titulo_resultado.configure(text=valor)
            self.mensagem.configure(
                text=(
                    "Essa é a classe indicada pelo modelo.\n\n"
                    "A previsão pode conter erros e não representa "
                    "uma verificação de autenticidade da moeda."
                )
            )
            print(
                f"Imagem: {self.nome_atual} | "
                f"Previsão: {valor} | Tempo: {tempo:.3f} s"
            )

        except Exception as erro:
            self.atualizar_status("ERRO NA ANÁLISE", "erro")
            self.titulo_resultado.configure(text="Não foi possível classificar")
            self.mensagem.configure(
                text="Verifique os arquivos do modelo e consulte o terminal."
            )
            print(f"\nErro na classificação: {erro}")
            messagebox.showerror(
                "Erro na classificação",
                "Não foi possível classificar a imagem.\n\n"
                f"Detalhes: {erro}",
            )

        finally:
            self.habilitar_classificacao(self.imagem_original is not None)

    def limpar(self):
        self.imagem_original = None
        self.nome_atual = None

        if self.previa_agendada is not None:
            try:
                self.after_cancel(self.previa_agendada)
            except tk.TclError:
                pass
            self.previa_agendada = None

        self.previa.configure(
            image="",
            text=TEXTO_PREVIA_VAZIA,
        )
        self.imagem_previa = None
        self.nome_arquivo.configure(text="Nenhuma imagem selecionada")
        self.atualizar_status("AGUARDANDO IMAGEM", "neutro")
        self.titulo_resultado.configure(text="Qual é o valor?")
        self.mensagem.configure(text="Selecione uma foto para começar.")
        self.habilitar_classificacao(False)

    # ========================================================
    # ABA 2 - DEMONSTRAÇÃO DE PDI
    # ========================================================

    def criar_aba_pdi(self):
        aba = self.aba_pdi
        aba.grid_columnconfigure(0, weight=1)
        aba.grid_rowconfigure(1, weight=1)

        topo = ctk.CTkFrame(aba, fg_color="transparent")
        topo.grid(row=0, column=0, sticky="ew")
        topo.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            topo,
            text="Demonstração do processamento de imagens",
            text_color=TEXTO,
            font=self.fonte_texto(21, True),
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 3))

        descricao = ctk.CTkLabel(
            topo,
            text=(
                "Exemplos previamente processados com anotações no Labelme "
                "e segmentação pelo algoritmo GrabCut. Esta demonstração é "
                "independente da classificação de uma nova imagem."
            ),
            text_color=SECUNDARIO,
            font=self.fonte_texto(13),
            justify="left",
            anchor="w",
            wraplength=700,
        )
        descricao.grid(row=1, column=0, sticky="w", padx=8, pady=(2, 12))
        self.ajustar_quebra(descricao, topo, 30)

        self.area_galeria = ctk.CTkScrollableFrame(
            aba,
            fg_color="transparent",
            corner_radius=16,
            scrollbar_button_color="#C5D2E3",
            scrollbar_button_hover_color="#9FB3CE",
        )
        self.area_galeria.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.area_galeria.grid_columnconfigure(0, weight=1)

    def carregar_galeria_pdi(self):
        # Limpa conteúdo anterior para permitir recarregar a galeria.
        for widget in self.area_galeria.winfo_children():
            widget.destroy()
        self.imagens_pdi.clear()

        if not PASTA_PDI.is_dir():
            self.mostrar_aviso_pdi(
                f"A pasta de resultados de PDI não foi encontrada:\n{PASTA_PDI}"
            )
            return

        # O processar_moeda.py salva os arquivos dentro de subpastas:
        # resultados_pdi/<classe>/<nome_do_exemplo>/
        comparacoes = sorted(PASTA_PDI.rglob("comparacao_pdi.png"))

        # Compatibilidade com a organização em que os arquivos estejam
        # diretamente na pasta da classe ou na raiz de resultados_pdi.
        if not comparacoes:
            originais = sorted(PASTA_PDI.rglob("original.png"))
            comparacoes = originais

        if not comparacoes:
            self.mostrar_aviso_pdi(
                "Nenhum exemplo de PDI foi encontrado.\n\n"
                "Confira se os resultados foram gerados pelo "
                "processar_moeda.py e salvos em resultados_pdi."
            )
            return

        for indice, caminho in enumerate(comparacoes):
            if caminho.name == "comparacao_pdi.png":
                pasta_exemplo = caminho.parent
                caminho_original = pasta_exemplo / "original.png"
                caminho_mascara = pasta_exemplo / "mascara.png"
                caminho_isolada = pasta_exemplo / "moeda_isolada.png"
            else:
                # Quando não existe comparacao_pdi.png, tenta montar
                # a visualização com os três arquivos individuais.
                pasta_exemplo = caminho.parent
                caminho_original = caminho
                caminho_mascara = pasta_exemplo / "mascara.png"
                caminho_isolada = pasta_exemplo / "moeda_isolada.png"

            if not all(
                arquivo.is_file()
                for arquivo in (caminho_original, caminho_mascara, caminho_isolada)
            ):
                continue

            relativo = pasta_exemplo.relative_to(PASTA_PDI)
            partes = relativo.parts
            classe = partes[0] if partes else "Exemplo"
            nome_exemplo = partes[-1] if len(partes) > 1 else "Exemplo"

            self.adicionar_cartao_pdi(
                indice + 1,
                classe,
                nome_exemplo,
                caminho_original,
                caminho_mascara,
                caminho_isolada,
            )

        if not self.area_galeria.winfo_children():
            self.mostrar_aviso_pdi(
                "Foram encontrados arquivos, mas faltam imagens correspondentes "
                "(original.png, mascara.png e moeda_isolada.png) em alguma pasta."
            )

    def mostrar_aviso_pdi(self, mensagem):
        ctk.CTkLabel(
            self.area_galeria,
            text=mensagem,
            text_color=SECUNDARIO,
            font=self.fonte_texto(15),
            justify="left",
            wraplength=720,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=20)

    def adicionar_cartao_pdi(
        self,
        indice,
        classe,
        nome_exemplo,
        caminho_original,
        caminho_mascara,
        caminho_isolada,
    ):
        cartao = ctk.CTkFrame(
            self.area_galeria,
            fg_color=PAINEL,
            corner_radius=18,
            border_width=1,
            border_color=BORDA,
        )
        cartao.grid(
            row=indice - 1,
            column=0,
            sticky="ew",
            padx=6,
            pady=8,
        )
        cartao.grid_columnconfigure((0, 1, 2), weight=1)

        titulo = f"{classe.replace('_', ' ').title()} — {nome_exemplo}"
        ctk.CTkLabel(
            cartao,
            text=titulo,
            text_color=TEXTO,
            font=self.fonte_texto(16, True),
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 8))

        arquivos = [
            ("1  •  Imagem original", caminho_original),
            ("2  •  Máscara binária", caminho_mascara),
            ("3  •  Moeda isolada", caminho_isolada),
        ]

        for coluna, (rotulo, caminho) in enumerate(arquivos):
            ctk.CTkLabel(
                cartao,
                text=rotulo,
                text_color=SECUNDARIO,
                font=self.fonte_texto(13, True),
            ).grid(row=1, column=coluna, padx=8, pady=(2, 5))

            try:
                with Image.open(caminho) as arquivo:
                    imagem = arquivo.convert("RGB")
                    imagem.thumbnail((230, 170))
                    imagem_ctk = ctk.CTkImage(
                        light_image=imagem,
                        size=imagem.size,
                    )
                self.imagens_pdi.append(imagem_ctk)

                ctk.CTkLabel(
                    cartao,
                    text="",
                    image=imagem_ctk,
                    fg_color=PREVIA,
                    corner_radius=12,
                    width=230,
                    height=180,
                ).grid(
                    row=2,
                    column=coluna,
                    sticky="nsew",
                    padx=8,
                    pady=(0, 12),
                )

            except (OSError, ValueError) as erro:
                ctk.CTkLabel(
                    cartao,
                    text=f"Não foi possível abrir a imagem:\n{erro}",
                    text_color="#A32626",
                    wraplength=200,
                ).grid(row=2, column=coluna, padx=8, pady=8)


if __name__ == "__main__":
    app = Aplicacao()
    app.mainloop()
