from pathlib import Path
from collections import Counter
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

PASTA = Path(__file__).resolve().parent

REFERENCIA = PASTA / "Merged35.csv"
ENTRADA = PASTA / "planilhas_mistas"
SAIDA = PASTA / "dataset_30_features"

SAIDA.mkdir(parents=True, exist_ok=True)

SEMENTE = 42
TAMANHO_BLOCO = 50000

MAX_AMOSTRA_AJUSTE = 200000

GERAR_SUBCONJUNTOS = True

CAMPOS_REMOVER = {
    "AVG",
    "LLC",
    "ARP",
    "Number",
    "Tot size",
    "fin_flag_number",
    "ack_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "is_single_packet"
}

ROTULOS = {"Label", "is_attack"}

BENIGNOS = {
    "BENIGN",
    "BENIGNTRAFFIC",
    "NORMAL",
    "NORMALTRAFFIC",
    "LEGITIMATE"
}

if not REFERENCIA.is_file():
    raise FileNotFoundError(
        f"Referencia nao encontrada: {REFERENCIA}"
    )

cabecalho = pd.read_csv(
    REFERENCIA, nrows=0
).columns.tolist()

features = [
    c for c in cabecalho
    if c not in CAMPOS_REMOVER
    and c not in ROTULOS
]

if len(features) != 30:
    raise ValueError(
        "Esperadas 30 caracteristicas no Merged35. "
        f"Encontradas: {len(features)}"
    )

print("\nCARACTERISTICAS SELECIONADAS")
for i, coluna in enumerate(features, start=1):
    print(f"{i:02d}. {coluna}")

arquivos = sorted(
    ENTRADA.rglob("*.csv")
)

if not arquivos:
    raise FileNotFoundError(
        "Nenhum arquivo CSV em planilhas_mistas."
    )

for arquivo in arquivos:

    cols = pd.read_csv(
        arquivo, nrows=0
    ).columns.tolist()

    faltantes = sorted(
        set(features) - set(cols)
    )

    extras = sorted(
        set(cols) -
        set(features) -
        CAMPOS_REMOVER -
        ROTULOS
    )

    if "Label" not in cols:
        raise ValueError(
            f"{arquivo.name}: coluna Label ausente."
        )

    if faltantes or extras:
        raise ValueError(
            f"\nArquivo: {arquivo.name}"
            f"\nColunas faltantes: {faltantes}"
            f"\nColunas extras: {extras}"
            "\nRevise antes de excluir outras colunas."
        )

print("\nArquivos encontrados:", len(arquivos))

def normalizar_label(serie):

    valores = (
        serie.astype("string")
        .str.strip()
    )

    if valores.isna().any() or valores.eq("").any():
        raise ValueError(
            "Existem rotulos ausentes ou vazios."
        )

    normalizados = (
        valores.str.upper()
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True
        )
    )

    return valores, normalizados


def iterar_blocos():

    colunas_necessarias = (
        set(features) | ROTULOS
    )

    for arquivo in arquivos:

        leitor = pd.read_csv(
            arquivo,
            chunksize=TAMANHO_BLOCO,
            low_memory=False,
            usecols=lambda c:
                c in colunas_necessarias
        )

        for bloco in leitor:

            # Exatamente as 30 features.
            X = bloco[features].apply(
                pd.to_numeric,
                errors="coerce"
            ).astype(np.float64)

            X = X.replace(
                [np.inf, -np.inf],
                np.nan
            )

            labels, labels_norm = normalizar_label(
                bloco["Label"]
            )

            benigno = (
                labels_norm.isin(BENIGNOS)
                .to_numpy(dtype=bool)
            )

            # Conferir is_attack caso exista.
            if "is_attack" in bloco.columns:

                valores_flag = (
                    bloco["is_attack"]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                )

                mapa = {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                }

                flag = valores_flag.map(mapa)

                if flag.isna().any():
                    raise ValueError(
                        f"{arquivo.name}: "
                        "is_attack possui valores invalidos."
                    )

                divergencias = (
                    flag.to_numpy(dtype=bool)
                    != (~benigno)
                )

                if divergencias.any():
                    raise ValueError(
                        f"{arquivo.name}: Label e "
                        "is_attack apresentam divergencias."
                    )

            hashes = (
                pd.util.hash_pandas_object(
                    X,
                    index=False
                ).to_numpy(dtype=np.uint64)
            )

            grupos = hashes % np.uint64(100)

            split = np.where(
                grupos < 70,
                "train",
                np.where(
                    grupos < 85,
                    "validation",
                    "test"
                )
            )

            yield arquivo, X, labels, benigno, split

print("\nAJUSTANDO PREPROCESSADOR...")

rng = np.random.default_rng(SEMENTE)

amostra_X = np.empty(
    (0, len(features)),
    dtype=np.float64
)

amostra_chaves = np.empty(
    0, dtype=np.float64
)

total_treino_benigno = 0
validos_por_coluna = np.zeros(
    len(features),
    dtype=np.int64
)

for arquivo, X, labels, benigno, split in iterar_blocos():

    mascara = (
        (split == "train") &
        benigno
    )

    X_benigno = X.loc[
        mascara
    ].to_numpy(dtype=np.float64)

    if len(X_benigno) == 0:
        continue

    total_treino_benigno += len(X_benigno)

    validos_por_coluna += (
        np.isfinite(X_benigno)
        .sum(axis=0)
    )

    novas_chaves = rng.random(
        len(X_benigno)
    )

    candidatos = np.concatenate(
        [amostra_X, X_benigno],
        axis=0
    )

    chaves = np.concatenate(
        [amostra_chaves, novas_chaves]
    )

    if len(chaves) > MAX_AMOSTRA_AJUSTE:

        selecionados = np.argpartition(
            chaves,
            MAX_AMOSTRA_AJUSTE - 1
        )[:MAX_AMOSTRA_AJUSTE]

        amostra_X = candidatos[
            selecionados
        ]

        amostra_chaves = chaves[
            selecionados
        ]

    else:
        amostra_X = candidatos
        amostra_chaves = chaves

if total_treino_benigno == 0:
    raise ValueError(
        "Nenhum registro benigno no treinamento."
    )

colunas_vazias = [
    features[i]
    for i, n in enumerate(validos_por_coluna)
    if n == 0
]

if colunas_vazias:
    raise ValueError(
        "Colunas sem valores no treino benigno: "
        f"{colunas_vazias}"
    )

preprocessador = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        RobustScaler()
    )
])

preprocessador.fit(
    pd.DataFrame(
        amostra_X,
        columns=features
    )
)

joblib.dump(
    preprocessador,
    SAIDA / "normalizador_30.joblib"
)

print(
    "Benignos de treino:",
    total_treino_benigno
)

print(
    "Registros usados para ajustar scaler:",
    len(amostra_X)
)

print("\nGERANDO DATASETS...")

escritos = set()
contagens = Counter()
total_linhas = 0
valores_ausentes = 0


def escrever(nome, dados):

    if dados.empty:
        return

    destino = SAIDA / nome
    primeira_vez = nome not in escritos

    dados.to_csv(
        destino,
        index=False,
        mode="w" if primeira_vez else "a",
        header=primeira_vez
    )

    escritos.add(nome)


for arquivo, X, labels, benigno, split in iterar_blocos():

    valores_ausentes += int(
        X.isna().sum().sum()
    )

    X_pronto = preprocessador.transform(X)

    X_pronto = np.asarray(
        X_pronto,
        dtype=np.float32
    )

    if not np.isfinite(X_pronto).all():
        raise ValueError(
            "Normalizacao produziu valores invalidos."
        )

    dados = pd.DataFrame(
        X_pronto,
        columns=features
    )

    dados["split"] = split

    rotulos = pd.DataFrame({
        "Label": labels.to_numpy(),
        "is_attack": (~benigno).astype(int),
        "split": split,
        "Arquivo_origem": str(
            arquivo.relative_to(ENTRADA)
        )
    })

    escrever(
        "dataset_preparado.csv",
        dados
    )

    escrever(
        "dataset_rotulos.csv",
        rotulos
    )

    if GERAR_SUBCONJUNTOS:

        m_treino = (
            (split == "train") & benigno
        )

        m_val_benigno = (
            (split == "validation") & benigno
        )

        m_val = split == "validation"
        m_teste = split == "test"

        escrever(
            "dataset_treino_benigno.csv",
            dados.loc[
                m_treino, features
            ]
        )

        escrever(
            "dataset_validacao_benigno.csv",
            dados.loc[
                m_val_benigno, features
            ]
        )

        escrever(
            "dataset_validacao_misto.csv",
            dados.loc[
                m_val, features
            ]
        )

        escrever(
            "rotulos_validacao_misto.csv",
            rotulos.loc[m_val]
        )

        escrever(
            "dataset_teste_misto.csv",
            dados.loc[
                m_teste, features
            ]
        )

        escrever(
            "rotulos_teste_misto.csv",
            rotulos.loc[m_teste]
        )

    grupos = pd.DataFrame({
        "Label": labels.to_numpy(),
        "split": split
    }).groupby(
        ["Label", "split"]
    ).size()

    for (classe, conjunto), quantidade in grupos.items():

        contagens[
            (str(classe), str(conjunto))
        ] += int(quantidade)

    total_linhas += len(dados)

    print(
        arquivo.name,
        "- registros processados:",
        total_linhas
    )

relatorio = pd.DataFrame([
    {
        "Label": classe,
        "split": conjunto,
        "Quantidade": quantidade
    }
    for (classe, conjunto), quantidade
    in sorted(contagens.items())
])

relatorio.to_csv(
    SAIDA / "relatorio_preprocessamento.csv",
    index=False
)

metadados = {
    "features": features,
    "numero_features": len(features),
    "colunas_removidas": sorted(
        CAMPOS_REMOVER
    ),
    "rotulos_separados": sorted(ROTULOS),
    "semente": SEMENTE,
    "divisao": {
        "train": 70,
        "validation": 15,
        "test": 15
    },
    "metodo_divisao":
        "hash deterministico das 30 features",
    "normalizacao": "RobustScaler",
    "imputacao": "mediana",
    "ajuste_normalizacao":
        "amostra de registros benignos de treino",
    "amostra_ajuste": len(amostra_X),
    "benignos_treino": total_treino_benigno,
    "linhas_processadas": total_linhas,
    "valores_ausentes_tratados": valores_ausentes,
    "arquivos_origem": [
        str(p.relative_to(ENTRADA))
        for p in arquivos
    ]
}

with open(
    SAIDA / "metadados.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadados,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 55)
print("PRE-PROCESSAMENTO CONCLUIDO")
print("Arquivos lidos:", len(arquivos))
print("Registros:", total_linhas)
print("Caracteristicas:", len(features))
print("Saida:", SAIDA)

print("\nDISTRIBUICAO DOS CONJUNTOS")

print(
    relatorio.groupby("split")[
        "Quantidade"
    ].sum().to_string()
)
