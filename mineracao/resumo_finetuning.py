from pathlib import Path
import ast
import csv
import json
import re


BASE = Path(
    "mineracao/resultados/resnet_s5"
)

EXPERIMENTOS = {
    "congelado": (
        BASE
        / "congelado"
        / "log_treino.txt"
    ),

    "layer4": (
        BASE
        / "layer4"
        / "log_treino.txt"
    ),

    "layer3_layer4": (
        BASE
        / "layer3_layer4"
        / "log_treino.txt"
    ),
}


def ler_texto(caminho):

    bruto = caminho.read_bytes()

    try:
        return bruto.decode("utf-8")

    except UnicodeDecodeError:
        return bruto.decode("cp1252")


def ultimo(padrao, texto):

    encontrados = re.findall(
        padrao,
        texto
    )

    if not encontrados:
        raise ValueError(
            f"Padrão não encontrado: {padrao}"
        )

    return encontrados[-1]


def ler_log(nome, caminho):

    if not caminho.exists():
        raise FileNotFoundError(
            caminho
        )

    texto = ler_texto(
        caminho
    )

    configuracao = ultimo(
        r"configuracao_finetuning=(\S+)",
        texto
    )

    arquitetura = ultimo(
        r"arquitetura=(\S+)",
        texto
    )

    parametros = int(
        ultimo(
            r"parametros_treinaveis=(\d+)",
            texto
        )
    )

    melhor_epoca = int(
        ultimo(
            r"melhor_epoca_validacao=(\d+)",
            texto
        )
    )

    lr_cabeca, lr_backbone = ultimo(
        r"lr_cabeca=([0-9.eE+-]+) "
        r"lr_backbone=([0-9.eE+-]+)",
        texto
    )

    threshold = ast.literal_eval(
        ultimo(
            r"threshold_validacao=(\{[^\n]+\})",
            texto
        )
    )

    metricas = ast.literal_eval(
        ultimo(
            r"metricas_validacao=(\{[^\n]+\})",
            texto
        )
    )

    sha = ultimo(
        r"npz_sha256=([0-9a-fA-F]{64})",
        texto
    )

    n_treino, n_validacao, n_teste = ultimo(
        r"n treino=(\d+) "
        r"validacao=(\d+) "
        r"teste=(\d+)",
        texto
    )

    epocas = [
        int(valor)
        for valor
        in re.findall(
            r"epoca=(\d+)",
            texto
        )
    ]

    return {
        "nome": nome,
        "configuracao": configuracao,
        "arquitetura": arquitetura,
        "parametros_treinaveis": parametros,
        "epocas_executadas": max(epocas),
        "melhor_epoca": melhor_epoca,
        "lr_cabeca": float(lr_cabeca),
        "lr_backbone": float(lr_backbone),
        "threshold": float(
            threshold["threshold"]
        ),
        "recall": float(
            metricas["recall"]
        ),
        "especificidade": float(
            metricas["especificidade"]
        ),
        "f1": float(
            metricas["f1"]
        ),
        "auc": float(
            metricas["auc"]
        ),
        "tp": int(
            metricas["tp"]
        ),
        "tn": int(
            metricas["tn"]
        ),
        "fp": int(
            metricas["fp"]
        ),
        "fn": int(
            metricas["fn"]
        ),
        "npz_sha256": sha,
        "n_treino": int(n_treino),
        "n_validacao": int(n_validacao),
        "n_teste": int(n_teste),
        "log_origem": str(caminho),
    }


def main():

    resultados = []

    for nome, caminho in EXPERIMENTOS.items():

        resultado = ler_log(
            nome,
            caminho
        )

        resultados.append(
            resultado
        )

    # --------------------------------------------------------
    # GARANTE QUE TODOS USARAM EXATAMENTE OS MESMOS DADOS
    # --------------------------------------------------------

    hashes = {
        resultado["npz_sha256"]
        for resultado in resultados
    }

    if len(hashes) != 1:
        raise ValueError(
            "Os experimentos usaram NPZs diferentes."
        )

    splits = {
        (
            resultado["n_treino"],
            resultado["n_validacao"],
            resultado["n_teste"],
        )
        for resultado in resultados
    }

    if len(splits) != 1:
        raise ValueError(
            "Os experimentos usaram splits diferentes."
        )

    # --------------------------------------------------------
    # ESCOLHA
    # --------------------------------------------------------
    #
    # O protocolo do projeto define recall da classe maligna
    # como métrica principal.
    #
    # Não usamos nenhuma informação do teste aqui.
    # --------------------------------------------------------

    maior_recall = max(
        resultado["recall"]
        for resultado in resultados
    )

    vencedores = [
        resultado
        for resultado in resultados
        if abs(
            resultado["recall"]
            - maior_recall
        ) < 1e-12
    ]

    if len(vencedores) != 1:
        raise ValueError(
            "Empate no recall. "
            "É necessário definir um critério "
            "de desempate antes de prosseguir."
        )

    vencedor = vencedores[0]

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    caminho_csv = (
        BASE
        / "resumo_validacao.csv"
    )

    campos = [
        "nome",
        "configuracao",
        "arquitetura",
        "parametros_treinaveis",
        "epocas_executadas",
        "melhor_epoca",
        "lr_cabeca",
        "lr_backbone",
        "threshold",
        "recall",
        "especificidade",
        "f1",
        "auc",
        "tp",
        "tn",
        "fp",
        "fn",
        "npz_sha256",
        "n_treino",
        "n_validacao",
        "n_teste",
        "log_origem",
    ]

    with caminho_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        escritor = csv.DictWriter(
            arquivo,
            fieldnames=campos
        )

        escritor.writeheader()

        escritor.writerows(
            resultados
        )

    # --------------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------------

    caminho_md = (
        BASE
        / "resumo_validacao.md"
    )

    linhas = [
        "# Comparação das configurações de fine-tuning",
        "",
        "Resultados extraídos automaticamente dos logs.",
        "",
        "| Configuração | Parâmetros treináveis | Melhor época | Recall | Especificidade | F1 | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for resultado in resultados:

        linhas.append(
            "| "
            f"{resultado['nome']} | "
            f"{resultado['parametros_treinaveis']} | "
            f"{resultado['melhor_epoca']} | "
            f"{resultado['recall']:.4f} | "
            f"{resultado['especificidade']:.4f} | "
            f"{resultado['f1']:.4f} | "
            f"{resultado['auc']:.4f} |"
        )

    linhas.extend(
        [
            "",
            "## Configuração escolhida",
            "",
            (
                "Critério de seleção: maior recall da "
                "classe maligna na validação."
            ),
            "",
            (
                f"Configuração escolhida: "
                f"`{vencedor['nome']}`."
            ),
            "",
            (
                "Nenhuma informação do conjunto de teste "
                "foi utilizada nesta escolha."
            ),
        ]
    )

    caminho_md.write_text(
        "\n".join(linhas) + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # JSON DA ESCOLHA
    # --------------------------------------------------------

    caminho_json = (
        BASE
        / "escolha_validacao.json"
    )

    escolha = {
        "criterio": (
            "maior recall da classe maligna "
            "na validacao"
        ),
        "teste_utilizado_na_escolha": False,
        "configuracao_escolhida": vencedor,
        "npz_sha256": vencedor["npz_sha256"],
        "experimentos_considerados": [
            resultado["nome"]
            for resultado in resultados
        ],
    }

    caminho_json.write_text(
        json.dumps(
            escolha,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print(
        "Resumo gerado com sucesso."
    )

    print(
        "CSV:",
        caminho_csv
    )

    print(
        "Markdown:",
        caminho_md
    )

    print(
        "JSON:",
        caminho_json
    )

    print()

    print(
        "Configuração escolhida pela validação:",
        vencedor["nome"]
    )

    print(
        "Recall:",
        vencedor["recall"]
    )

    print(
        "Especificidade:",
        vencedor["especificidade"]
    )

    print(
        "AUC:",
        vencedor["auc"]
    )

    print()

    print(
        "Nenhum dado de teste foi usado nesta escolha."
    )


if __name__ == "__main__":
    main()