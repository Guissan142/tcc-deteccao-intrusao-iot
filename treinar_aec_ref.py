
import os
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import json
import argparse
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

PASTA = Path(__file__).resolve().parent

ARQ_TREINO = PASTA / "dataset_30_features/dataset_treino_benigno.csv"
ARQ_VALIDACAO = PASTA / "dataset_30_features/dataset_validacao_benigno.csv"
ARQ_TESTE = PASTA / "dataset_30_features/dataset_teste_misto.csv"
ARQ_ROTULOS = PASTA / "dataset_30_features/rotulos_teste_misto.csv"
ARQ_METADADOS = PASTA / "dataset_30_features/metadados.json"

SAIDA = PASTA / "resultados_aec_ref"
SAIDA.mkdir(parents=True, exist_ok=True)

EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 0.001
PERCENTIL = 90

SEMENTE_DIVISAO_VALIDACAO = 2026

COLUNAS_METADATA = {
    "split",
    "Label",
    "is_attack",
    "Unnamed: 0",
}

CLASSES_BENIGNAS = {
    "BENIGN",
    "BENIGNTRAFFIC",
    "BENIGN_TRAFFIC",
    "NORMAL",
    "0",
    "FALSE",
}

def ler_csv(caminho):
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho}"
        )

    return pd.read_csv(caminho)


def obter_features(dados_treino):
    if ARQ_METADADOS.exists():

        with open(
            ARQ_METADADOS,
            "r",
            encoding="utf-8"
        ) as arquivo:
            metadados = json.load(arquivo)

        features = metadados.get("features")

        if not isinstance(features, list):
            raise ValueError(
                "O metadados.json não contém uma lista "
                "'features' válida."
            )

    else:
        print(
            "AVISO: metadados.json não encontrado. "
            "Será utilizada a ordem do CSV de treino."
        )

        features = [
            c for c in dados_treino.columns
            if c not in COLUNAS_METADATA
            and not c.startswith("Unnamed:")
        ]

    if len(features) != 30:
        raise ValueError(
            f"Esperadas 30 features, encontradas "
            f"{len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            "Existem nomes duplicados nas features."
        )

    return features


def verificar_benignos(dados, nome):
    if "is_attack" in dados.columns:

        valores = pd.to_numeric(
            dados["is_attack"],
            errors="raise"
        )

        if not (valores == 0).all():
            raise ValueError(
                f"{nome} contém registros identificados "
                "como ataques."
            )

    if "Label" in dados.columns:

        valores = (
            dados["Label"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        if not valores.isin(CLASSES_BENIGNAS).all():
            raise ValueError(
                f"{nome} contém rótulos não benignos."
            )


def preparar_x(dados, features, nome):
    faltantes = [
        c for c in features
        if c not in dados.columns
    ]

    if faltantes:
        raise ValueError(
            f"{nome}: features ausentes: {faltantes}"
        )

    extras = [
        c for c in dados.columns
        if c not in features
        and c not in COLUNAS_METADATA
        and not c.startswith("Unnamed:")
    ]

    if extras:
        raise ValueError(
            f"{nome}: colunas inesperadas: {extras}"
        )

    x = (
        dados.loc[:, features]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(x).all():
        raise ValueError(
            f"{nome}: existem NaN ou infinitos."
        )

    return x

def preparar_rotulos(dados):
    if "Label" in dados.columns:
        classes = dados["Label"]

    elif "is_attack" in dados.columns:
        classes = dados["is_attack"]

    elif dados.shape[1] == 1:
        classes = dados.iloc[:, 0]

    else:
        raise ValueError(
            "Coluna de rótulos não identificada."
        )

    if classes.isna().any():
        raise ValueError("Existem rótulos ausentes.")

    classes = classes.astype(str).str.strip()

    y = (
        ~classes.str.upper().isin(CLASSES_BENIGNAS)
    ).astype(np.int32).to_numpy()

    return y, classes.to_numpy()

def dividir_validacao(x_validacao):
    rng = np.random.default_rng(
        SEMENTE_DIVISAO_VALIDACAO
    )

    indices = rng.permutation(
        len(x_validacao)
    )

    n_early = int(len(indices) * 0.40)
    n_calibracao = int(len(indices) * 0.30)

    idx_early = indices[:n_early]

    idx_calibracao = indices[
        n_early:n_early + n_calibracao
    ]

    idx_avaliacao = indices[
        n_early + n_calibracao:
    ]

    if min(
        len(idx_early),
        len(idx_calibracao),
        len(idx_avaliacao)
    ) == 0:
        raise ValueError(
            "Validação benigna insuficiente."
        )

    np.savez(
        SAIDA / "indices_validacao.npz",
        early=idx_early,
        calibracao=idx_calibracao,
        avaliacao=idx_avaliacao,
    )

    return (
        x_validacao[idx_early],
        x_validacao[idx_calibracao],
        x_validacao[idx_avaliacao],
    )

def criar_modelo(n_features):
    entrada = keras.Input(
        shape=(n_features,),
        name="entrada"
    )

    x = keras.layers.Dense(
        32,
        activation="relu"
    )(entrada)

    x = keras.layers.Dense(
        16,
        activation="relu"
    )(x)

    x = keras.layers.Dense(
        6,
        activation="relu",
        name="latente"
    )(x)

    x = keras.layers.Dense(
        16,
        activation="relu"
    )(x)

    x = keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    saida = keras.layers.Dense(
        n_features,
        activation="linear"
    )(x)

    modelo = keras.Model(
        inputs=entrada,
        outputs=saida,
        name="AEC_REF"
    )

    modelo.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse"
    )

    if modelo.count_params() != 3268:
        raise ValueError(
            "Quantidade inesperada de parâmetros."
        )

    return modelo

def calcular_erros(modelo, x):
    reconstruido = modelo.predict(
        x,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    erros = np.mean(
        np.square(x - reconstruido),
        axis=1
    )

    return erros

def treinar_semente(
    semente,
    x_treino,
    x_early,
    x_calibracao,
    x_avaliacao
):
    print("\n" + "=" * 60)
    print(f"TREINAMENTO AEC-REF | SEMENTE {semente}")
    print("=" * 60)

    keras.backend.clear_session()
    keras.utils.set_random_seed(semente)

    modelo = criar_modelo(
        x_treino.shape[1]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        )
    ]

    inicio = __import__("time").perf_counter()

    historico = modelo.fit(
        x_treino,
        x_treino,
        validation_data=(
            x_early,
            x_early
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
        verbose=2
    )

    tempo_treino = (
        __import__("time").perf_counter() - inicio
    )

    erros_calibracao = calcular_erros(
        modelo,
        x_calibracao
    )

    limiar = float(
        np.percentile(
            erros_calibracao,
            PERCENTIL
        )
    )

    erros_avaliacao = calcular_erros(
        modelo,
        x_avaliacao
    )

    fpr_benigna = float(
        np.mean(erros_avaliacao > limiar)
    )

    pasta_semente = (
        SAIDA / f"seed_{semente}"
    )

    pasta_semente.mkdir(
        parents=True,
        exist_ok=True
    )

    caminho_modelo = (
        pasta_semente / "AEC_REF.keras"
    )

    modelo.save(caminho_modelo)

    pd.DataFrame(
        historico.history
    ).to_csv(
        pasta_semente / "historico.csv",
        index=False
    )

    configuracao = {
        "modelo": "AEC-REF",
        "semente": semente,
        "arquitetura": [
            30, 32, 16, 6, 16, 32, 30
        ],
        "parametros": modelo.count_params(),
        "funcao_perda": "MSE",
        "otimizador": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs_max": EPOCHS,
        "epochs_executadas": len(
            historico.history["loss"]
        ),
        "melhor_val_loss": float(
            min(historico.history["val_loss"])
        ),
        "percentil": PERCENTIL,
        "limiar": limiar,
        "fpr_validacao_benigna": fpr_benigna,
        "tempo_treinamento_s": tempo_treino,
        "tensorflow": tf.__version__,
        "python": platform.python_version(),
    }

    with open(
        pasta_semente / "configuracao.json",
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            configuracao,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    print("\nTREINAMENTO CONCLUÍDO")
    print("Épocas:", configuracao["epochs_executadas"])
    print("Menor val_loss:", configuracao["melhor_val_loss"])
    print("Limiar P90:", limiar)
    print("FPR benigna:", f"{fpr_benigna:.4%}")
    print("Tempo de treinamento:", f"{tempo_treino:.2f}s")
    print("Modelo salvo:", caminho_modelo)

    return modelo, configuracao

def avaliar_teste(modelo, limiar, features):
    teste = ler_csv(ARQ_TESTE)
    rotulos = ler_csv(ARQ_ROTULOS)

    x = preparar_x(
        teste,
        features,
        "teste"
    )

    y, classes = preparar_rotulos(
        rotulos
    )

    if len(x) != len(y):
        raise ValueError(
            "Número de registros e rótulos diferente."
        )

    erros = calcular_erros(modelo, x)

    pred = (erros > limiar).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    metricas = {
        "precisao": float(
            precision_score(
                y, pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, pred, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y, pred, zero_division=0
            )
        ),
        "acuracia": float(
            accuracy_score(y, pred)
        ),
        "fpr": float(fpr),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": len(y),
    }

    pasta_teste = SAIDA / "teste_seed_1992"
    pasta_teste.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        pasta_teste / "metricas.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            metricas,
            arquivo,
            indent=4
        )

    pd.DataFrame({
        "classe_real": classes,
        "rotulo_binario": y,
        "erro_reconstrucao": erros,
        "predicao": pred
    }).to_csv(
        pasta_teste / "predicoes.csv",
        index=False
    )

    resultados_classes = []

    for classe in np.unique(classes):
        mascara = (
            (classes == classe) &
            (y == 1)
        )

        if not mascara.any():
            continue

        resultados_classes.append({
            "classe": classe,
            "quantidade": int(mascara.sum()),
            "recall": float(
                pred[mascara].mean()
            )
        })

    pd.DataFrame(
        resultados_classes
    ).to_csv(
        pasta_teste / "recall_por_classe.csv",
        index=False
    )

    print("\nRESULTADOS DO TESTE")
    print("=" * 50)

    for nome, valor in metricas.items():
        print(f"{nome}: {valor}")

    return metricas

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sementes",
        nargs="+",
        type=int,
        default=[1992]
    )

    parser.add_argument(
        "--avaliar-teste",
        action="store_true"
    )

    args = parser.parse_args()

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as erro:
        print("Aviso sobre determinismo:", erro)

    treino = ler_csv(ARQ_TREINO)
    validacao = ler_csv(ARQ_VALIDACAO)

    verificar_benignos(
        treino,
        "treinamento"
    )

    verificar_benignos(
        validacao,
        "validação"
    )

    features = obter_features(treino)

    x_treino = preparar_x(
        treino,
        features,
        "treinamento"
    )

    x_validacao = preparar_x(
        validacao,
        features,
        "validação"
    )

    with open(
        SAIDA / "features_utilizadas.json",
        "w",
        encoding="utf-8"
    ) as arquivo:
        json.dump(
            features,
            arquivo,
            indent=4,
            ensure_ascii=False
        )

    x_early, x_cal, x_eval = (
        dividir_validacao(x_validacao)
    )

    print("\nDADOS CARREGADOS")
    print("Treinamento benigno:", len(x_treino))
    print("Validação / EarlyStopping:", len(x_early))
    print("Calibração P90:", len(x_cal))
    print("Avaliação benigna:", len(x_eval))
    print("Características:", len(features))

    resumo = []
    modelo_1992 = None
    limiar_1992 = None

    for semente in dict.fromkeys(args.sementes):
        modelo, config = treinar_semente(
            semente,
            x_treino,
            x_early,
            x_cal,
            x_eval
        )

        resumo.append(config)

        if semente == 1992:
            modelo_1992 = modelo
            limiar_1992 = config["limiar"]

    pd.DataFrame(resumo).to_csv(
        SAIDA / "resumo_sementes.csv",
        index=False
    )

    if args.avaliar_teste:

        if modelo_1992 is None:
            raise ValueError(
                "Para avaliar o teste, inclua "
                "a semente 1992."
            )

        avaliar_teste(
            modelo_1992,
            limiar_1992,
            features
        )

    print("\nPROCESSO FINALIZADO")
    print("Resultados em:", SAIDA)


if __name__ == "__main__":
    main()
