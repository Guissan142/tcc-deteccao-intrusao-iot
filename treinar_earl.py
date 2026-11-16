
from pathlib import Path
from itertools import zip_longest
from collections import defaultdict
import gc
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PASTA = Path(__file__).resolve().parent

DADOS = PASTA / "dataset_30_features"
SAIDA = PASTA / "resultados_earl"

SAIDA.mkdir(parents=True, exist_ok=True)

SEMENTES = [8, 42, 167, 800, 1992]

EPOCAS = 80
BATCH_SIZE = 512
BATCH_PREDICAO = 2048
CHUNK_AVALIACAO = 20000

LEARNING_RATE = 0.001
LR_MIN = 0.00001

PACIENCIA_PARADA = 8
PACIENCIA_LR = 3
MIN_DELTA = 1e-6

PERCENTIL = 90

BETA = 0.5

RUIDO_FIXO = 0.03

RUIDO_MIN = 0.005
RUIDO_MAX = 0.03
EPOCAS_AQUECIMENTO = 20

FPR_MAXIMA = 0.105

CONFIGURACOES = [
    {
        "nome": "AEC-REF",
        "ruido": "nenhum",
        "lambda": 0.0
    },
    {
        "nome": "AE-PE",
        "ruido": "fixo",
        "lambda": 0.0
    },
    {
        "nome": "AE-PECL-005",
        "ruido": "fixo",
        "lambda": 0.005
    },
    {
        "nome": "AE-PECL-010",
        "ruido": "fixo",
        "lambda": 0.01
    },
    {
        "nome": "AE-PECL-020",
        "ruido": "fixo",
        "lambda": 0.02
    },
    {
        "nome": "AE-PECL-PROG",
        "ruido": "progressivo",
        "lambda": 0.01
    }
]

CONTINUAS = {
    "Header_Length",
    "Time_To_Live",
    "Rate",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "Tot sum",
    "Min",
    "Max",
    "Std",
    "IAT",
    "Variance"
}

BENIGNOS = {
    "BENIGN",
    "BENIGNTRAFFIC",
    "NORMAL",
    "NORMALTRAFFIC",
    "LEGITIMATE"
}

arquivo_metadados = DADOS / "metadados.json"

with open(
    arquivo_metadados,
    "r",
    encoding="utf-8"
) as f:
    metadados = json.load(f)

FEATURES = metadados["features"]
N_FEATURES = len(FEATURES)

if N_FEATURES != 30:
    raise ValueError(
        f"Esperadas 30 features, obtidas {N_FEATURES}"
    )

if len(set(FEATURES)) != 30:
    raise ValueError("Features duplicadas.")

if not CONTINUAS.issubset(set(FEATURES)):
    raise ValueError(
        "Lista de features continuas incompativel."
    )

configuracao_experimento = {
    "versao": "EARL",
    "features": FEATURES,
    "sementes": SEMENTES,
    "percentil": PERCENTIL,
    "epocas": EPOCAS,
    "batch_size": BATCH_SIZE,
    "beta": BETA,
    "ruido_fixo": RUIDO_FIXO,
    "ruido_min": RUIDO_MIN,
    "ruido_max": RUIDO_MAX,
    "epocas_aquecimento": EPOCAS_AQUECIMENTO,
    "fpr_maxima": FPR_MAXIMA,
    "configuracoes": CONFIGURACOES
}

with open(
    SAIDA / "configuracao_earl.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        configuracao_experimento,
        f,
        indent=4,
        ensure_ascii=False
    )

def carregar_matriz(nome):

    caminho = DADOS / nome

    if not caminho.is_file():
        raise FileNotFoundError(caminho)

    print("Carregando:", nome)

    df = pd.read_csv(
        caminho,
        usecols=FEATURES,
        dtype={
            col: np.float32
            for col in FEATURES
        }
    )

    X = df[FEATURES].to_numpy(
        dtype=np.float32
    )

    if len(X) == 0:
        raise ValueError("Arquivo vazio: " + nome)

    if not np.isfinite(X).all():
        raise ValueError(
            "Valores invalidos: " + nome
        )

    return X


X_TREINO = carregar_matriz(
    "dataset_treino_benigno.csv"
)

X_VALIDACAO = carregar_matriz(
    "dataset_validacao_benigno.csv"
)

rng = np.random.default_rng(2026)

indices = rng.permutation(
    len(X_VALIDACAO)
)

n = len(indices)

corte1 = int(n * 0.40)
corte2 = int(n * 0.70)

ids_parada = indices[:corte1]
ids_calibracao = indices[corte1:corte2]
ids_avaliacao = indices[corte2:]

if not all([
    len(ids_parada) > 0,
    len(ids_calibracao) > 0,
    len(ids_avaliacao) > 0
]):
    raise ValueError(
        "Dados insuficientes para validacao."
    )

X_PARADA = X_VALIDACAO[ids_parada]
X_CALIBRACAO = X_VALIDACAO[ids_calibracao]

MASCARA_AVALIACAO = np.zeros(
    n,
    dtype=bool
)

MASCARA_AVALIACAO[ids_avaliacao] = True

print("\nDIVISAO DOS DADOS")
print("Treinamento:", len(X_TREINO))
print("Early Stopping:", len(X_PARADA))
print("Calibracao:", len(X_CALIBRACAO))
print("Benignos de avaliacao:",
      int(MASCARA_AVALIACAO.sum()))

escala_ruido = np.zeros(
    N_FEATURES,
    dtype=np.float32
)

for j, feature in enumerate(FEATURES):

    if feature not in CONTINUAS:
        continue

    coluna = X_TREINO[:, j].astype(
        np.float64
    )

    mediana = np.median(coluna)

    mad = np.median(
        np.abs(coluna - mediana)
    )

    escala_ruido[j] = np.clip(
        1.4826 * mad,
        0.1,
        2.0
    )

SIGMA = tf.constant(
    escala_ruido,
    dtype=tf.float32
)

pd.DataFrame({
    "Feature": FEATURES,
    "Escala_ruido": escala_ruido,
    "Perturbada": escala_ruido > 0
}).to_csv(
    SAIDA / "caracteristicas_perturbadas_earl.csv",
    index=False
)

def criar_autoencoder():

    entrada = keras.Input(
        shape=(N_FEATURES,)
    )

    z = layers.Dense(
        32,
        activation="relu"
    )(entrada)

    z = layers.Dense(
        16,
        activation="relu"
    )(z)

    z = layers.Dense(
        6,
        activation="relu"
    )(z)

    encoder = keras.Model(
        entrada,
        z,
        name="encoder"
    )

    entrada_latente = keras.Input(
        shape=(6,)
    )

    x = layers.Dense(
        16,
        activation="relu"
    )(entrada_latente)

    x = layers.Dense(
        32,
        activation="relu"
    )(x)

    saida = layers.Dense(
        N_FEATURES,
        activation="linear"
    )(x)

    decoder = keras.Model(
        entrada_latente,
        saida,
        name="decoder"
    )

    entrada_completa = keras.Input(
        shape=(N_FEATURES,)
    )

    saida_completa = decoder(
        encoder(entrada_completa)
    )

    modelo = keras.Model(
        entrada_completa,
        saida_completa,
        name="Autoencoder_Compacto"
    )

    if modelo.count_params() != 3268:
        raise ValueError(
            "Arquitetura diferente do AE05B."
        )

    return encoder, decoder, modelo

def calcular_erros(modelo, X):

    if len(X) == 0:
        return np.empty(0, dtype=np.float64)

    predicao = modelo.predict(
        X,
        batch_size=BATCH_PREDICAO,
        verbose=0
    )

    diferenca = (
        X.astype(np.float64)
        - predicao.astype(np.float64)
    )

    erros = np.mean(
        np.square(diferenca),
        axis=1
    )

    if not np.isfinite(erros).all():
        raise ValueError(
            "Erros de reconstrucao invalidos."
        )

    return erros

def obter_rotulos(bloco):

    if "Label" not in bloco.columns:
        raise ValueError(
            "Coluna Label ausente."
        )

    labels = (
        bloco["Label"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    if labels.isna().any():
        raise ValueError(
            "Existem rotulos ausentes."
        )

    if labels.eq("").any():
        raise ValueError(
            "Existem rotulos vazios."
        )

    normalizados = labels.str.replace(
        r"[^A-Z0-9]",
        "",
        regex=True
    )

    y = (
        ~normalizados.isin(BENIGNOS)
    ).to_numpy(dtype=np.int8)

    if "is_attack" in bloco.columns:

        flags = (
            bloco["is_attack"]
            .astype("string")
            .str.strip()
            .str.lower()
            .map({
                "0": 0,
                "1": 1,
                "false": 0,
                "true": 1
            })
        )

        if flags.isna().any():
            raise ValueError(
                "Valores invalidos em is_attack."
            )

        if not np.array_equal(
            flags.to_numpy(dtype=np.int8),
            y
        ):
            raise ValueError(
                "Inconsistencia Label/is_attack."
            )

    return labels.to_numpy(dtype=str), y

def calcular_metricas(tn, fp, fn, tp):

    total = tn + fp + fn + tp

    precisao = (
        tp / (tp + fp)
        if tp + fp else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0.0
    )

    f1 = (
        2 * precisao * recall
        / (precisao + recall)
        if precisao + recall else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn else 0.0
    )

    acuracia = (
        (tp + tn) / total
        if total else 0.0
    )

    return {
        "Precisao": precisao,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "Acuracia": acuracia,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ==================================================
# 10. AVALIAR VALIDACAO MISTA
# ==================================================

def avaliar_modelo(modelo, limiar):

    caminho_features = (
        DADOS / "dataset_validacao_misto.csv"
    )

    caminho_rotulos = (
        DADOS / "rotulos_validacao_misto.csv"
    )

    leitor_X = pd.read_csv(
        caminho_features,
        usecols=FEATURES,
        chunksize=CHUNK_AVALIACAO
    )

    leitor_y = pd.read_csv(
        caminho_rotulos,
        chunksize=CHUNK_AVALIACAO
    )

    tn = fp = fn = tp = 0

    contador_benignos = 0

    por_classe = defaultdict(
        lambda: [0, 0]
    )

    for bloco_X, bloco_y in zip_longest(
        leitor_X,
        leitor_y
    ):

        if bloco_X is None or bloco_y is None:
            raise ValueError(
                "Tamanhos de arquivos diferentes."
            )

        if len(bloco_X) != len(bloco_y):
            raise ValueError(
                "Blocos de features e rotulos desalinhados."
            )

        X = bloco_X[FEATURES].to_numpy(
            dtype=np.float32
        )

        if not np.isfinite(X).all():
            raise ValueError(
                "Features invalidas na avaliacao."
            )

        labels, y = obter_rotulos(
            bloco_y
        )

        # Retirar os benignos de Early Stopping
        # e calibracao do conjunto de avaliacao.

        pos_benignos = np.flatnonzero(
            y == 0
        )

        inicio = contador_benignos
        fim = inicio + len(pos_benignos)

        if fim > len(X_VALIDACAO):
            raise ValueError(
                "Quantidade inesperada de benignos."
            )

        if len(pos_benignos):

            referencia = X_VALIDACAO[
                inicio:fim
            ]

            if not np.allclose(
                X[pos_benignos],
                referencia,
                rtol=1e-5,
                atol=1e-6
            ):
                raise ValueError(
                    "Ordem dos benignos nao coincide."
                )

        manter = np.ones(
            len(X),
            dtype=bool
        )

        manter[pos_benignos] = (
            MASCARA_AVALIACAO[inicio:fim]
        )

        contador_benignos = fim

        X = X[manter]
        y = y[manter]
        labels = labels[manter]

        if len(X) == 0:
            continue

        erros = calcular_erros(
            modelo,
            X
        )

        pred = (
            erros > limiar
        ).astype(np.int8)

        tn += int(
            ((y == 0) & (pred == 0)).sum()
        )

        fp += int(
            ((y == 0) & (pred == 1)).sum()
        )

        fn += int(
            ((y == 1) & (pred == 0)).sum()
        )

        tp += int(
            ((y == 1) & (pred == 1)).sum()
        )

        for classe in np.unique(
            labels[y == 1]
        ):

            mascara = (
                (labels == classe)
                & (y == 1)
            )

            total_classe = int(
                mascara.sum()
            )

            detectados = int(
                pred[mascara].sum()
            )

            por_classe[classe][0] += (
                total_classe
            )

            por_classe[classe][1] += (
                detectados
            )

    if contador_benignos != len(X_VALIDACAO):
        raise ValueError(
            "Contagem final de benignos incorreta."
        )

    metricas = calcular_metricas(
        tn, fp, fn, tp
    )

    return metricas, por_classe


# ==================================================
# 11. FUNCAO DE RUIDO
# ==================================================

def intensidade_ruido(tipo, epoca):

    if tipo == "nenhum":
        return 0.0

    if tipo == "fixo":
        return RUIDO_FIXO

    if tipo == "progressivo":

        progresso = min(
            epoca / EPOCAS_AQUECIMENTO,
            1.0
        )

        return (
            RUIDO_MIN
            + (RUIDO_MAX - RUIDO_MIN)
            * progresso
        )

    raise ValueError(
        "Tipo de ruido desconhecido."
    )


# ==================================================
# 12. TREINAMENTO PERSONALIZADO
# ==================================================

def treinar(configuracao, semente):

    nome = configuracao["nome"]
    tipo_ruido = configuracao["ruido"]
    lamb = float(configuracao["lambda"])

    usa_ruido = tipo_ruido != "nenhum"
    beta = BETA if usa_ruido else 0.0

    keras.backend.clear_session()

    keras.utils.set_random_seed(
        semente
    )

    encoder, decoder, modelo = (
        criar_autoencoder()
    )

    otimizador = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    dados_tf = (
        tf.data.Dataset
        .from_tensor_slices(X_TREINO)
        .shuffle(
            min(20000, len(X_TREINO)),
            seed=semente,
            reshuffle_each_iteration=True
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # A regra de treinamento e definida uma vez
    # para cada modelo/semente.

    @tf.function
    def passo_treino(x, intensidade):

        with tf.GradientTape() as tape:

            z_limpo = encoder(
                x,
                training=True
            )

            reconstruido_limpo = decoder(
                z_limpo,
                training=True
            )

            perda_rec = tf.reduce_mean(
                tf.square(
                    x - reconstruido_limpo
                )
            )

            perda_denoise = tf.constant(
                0.0,
                dtype=tf.float32
            )

            perda_cons = tf.constant(
                0.0,
                dtype=tf.float32
            )

            if usa_ruido:

                ruido = (
                    tf.random.normal(
                        tf.shape(x),
                        dtype=tf.float32
                    )
                    * SIGMA
                    * intensidade
                )

                x_perturbado = x + ruido

                z_perturbado = encoder(
                    x_perturbado,
                    training=True
                )

                reconstruido_perturbado = decoder(
                    z_perturbado,
                    training=True
                )

                perda_denoise = tf.reduce_mean(
                    tf.square(
                        x - reconstruido_perturbado
                    )
                )

                if lamb > 0:

                    perda_cons = tf.reduce_mean(
                        tf.square(
                            z_limpo - z_perturbado
                        )
                    )

            perda_total = (
                perda_rec
                + beta * perda_denoise
                + lamb * perda_cons
            )

        variaveis = modelo.trainable_variables

        gradientes = tape.gradient(
            perda_total,
            variaveis
        )

        otimizador.apply_gradients(
            zip(
                gradientes,
                variaveis
            )
        )

        return (
            perda_total,
            perda_rec,
            perda_denoise,
            perda_cons
        )

    melhor_validacao = float("inf")
    melhores_pesos = None
    melhor_epoca = 0

    espera_parada = 0
    espera_lr = 0

    historico = []

    for epoca in range(1, EPOCAS + 1):

        intensidade = intensidade_ruido(
            tipo_ruido,
            epoca
        )

        intensidade_tf = tf.constant(
            intensidade,
            dtype=tf.float32
        )

        somas = np.zeros(
            4,
            dtype=np.float64
        )

        total_amostras = 0

        for lote in dados_tf:

            perdas = passo_treino(
                lote,
                intensidade_tf
            )

            tamanho_lote = int(
                tf.shape(lote)[0].numpy()
            )

            valores = np.array(
                [float(v.numpy()) for v in perdas],
                dtype=np.float64
            )

            somas += valores * tamanho_lote

            total_amostras += tamanho_lote

        medias = somas / total_amostras

        # O Early Stopping monitora a reconstrucao
        # de dados benignos sem perturbacao.
        val_loss = float(
            calcular_erros(
                modelo,
                X_PARADA
            ).mean()
        )

        lr_atual = float(
            otimizador.learning_rate.numpy()
        )

        historico.append({
            "Epoca": epoca,
            "Perda_total": medias[0],
            "Perda_rec": medias[1],
            "Perda_denoise": medias[2],
            "Perda_cons": medias[3],
            "MSE_validacao": val_loss,
            "Learning_rate": lr_atual,
            "Intensidade_ruido": intensidade,
            "Lambda": lamb,
            "Beta": beta
        })

        print(
            f"{nome} | seed={semente} "
            f"| epoca={epoca:03d} "
            f"| loss={medias[0]:.6f} "
            f"| val={val_loss:.6f} "
            f"| lr={lr_atual:.6f}"
        )

        if val_loss < (
            melhor_validacao - MIN_DELTA
        ):

            melhor_validacao = val_loss

            melhores_pesos = (
                modelo.get_weights()
            )

            melhor_epoca = epoca

            espera_parada = 0
            espera_lr = 0

        else:

            espera_parada += 1
            espera_lr += 1

            if espera_lr >= PACIENCIA_LR:

                novo_lr = max(
                    LR_MIN,
                    lr_atual * 0.5
                )

                otimizador.learning_rate.assign(
                    novo_lr
                )

                espera_lr = 0

            if espera_parada >= PACIENCIA_PARADA:

                print(
                    "Early Stopping na epoca:",
                    epoca
                )

                break

    if melhores_pesos is None:
        raise RuntimeError(
            "Nenhum conjunto de pesos salvo."
        )

    modelo.set_weights(
        melhores_pesos
    )

    return (
        modelo,
        pd.DataFrame(historico),
        melhor_epoca,
        melhor_validacao
    )


# ==================================================
# 13. EXECUTAR EXPERIMENTOS
# ==================================================

resultados = []
classes_resultados = []

total_execucoes = (
    len(CONFIGURACOES) * len(SEMENTES)
)

numero_execucao = 0

for configuracao in CONFIGURACOES:

    nome = configuracao["nome"]

    for semente in SEMENTES:

        numero_execucao += 1

        print("\n" + "=" * 65)

        print(
            f"EXECUCAO {numero_execucao}/"
            f"{total_execucoes}"
        )

        print("Modelo:", nome)
        print("Semente:", semente)
        print("Lambda:", configuracao["lambda"])
        print("Ruido:", configuracao["ruido"])

        modelo, historico, melhor_epoca, melhor_val = (
            treinar(
                configuracao,
                semente
            )
        )

        # ======================================
        # CALIBRAR P90 SOMENTE COM BENIGNOS
        # ======================================

        erros_calibracao = calcular_erros(
            modelo,
            X_CALIBRACAO
        )

        limiar = float(
            np.percentile(
                erros_calibracao,
                PERCENTIL
            )
        )

        # ======================================
        # AVALIAR CONJUNTO INDEPENDENTE
        # ======================================

        metricas, classes = avaliar_modelo(
            modelo,
            limiar
        )

        resultados.append({
            "Modelo": nome,
            "Semente": semente,
            "Tipo_ruido": configuracao["ruido"],
            "Beta": (
                BETA
                if configuracao["ruido"] != "nenhum"
                else 0.0
            ),
            "Lambda": configuracao["lambda"],
            "Percentil": PERCENTIL,
            "Limiar": limiar,
            "Parametros": modelo.count_params(),
            "Epocas": len(historico),
            "Melhor_epoca": melhor_epoca,
            "Melhor_MSE_validacao": melhor_val,
            **metricas
        })

        for classe, valores in classes.items():

            total = valores[0]
            detectados = valores[1]

            classes_resultados.append({
                "Modelo": nome,
                "Semente": semente,
                "Classe": classe,
                "Total": total,
                "Detectados": detectados,
                "FN": total - detectados,
                "Recall_classe": (
                    detectados / total
                    if total else 0.0
                )
            })

        # Salvar pesos e historico
        modelo.save(
            SAIDA /
            f"{nome}_seed_{semente}.keras"
        )

        historico.to_csv(
            SAIDA /
            f"historico_{nome}_seed_{semente}.csv",
            index=False
        )

        # Salvar progresso a cada execucao
        pd.DataFrame(
            resultados
        ).to_csv(
            SAIDA / "resultados_earl.csv",
            index=False
        )

        pd.DataFrame(
            classes_resultados
        ).to_csv(
            SAIDA / "classes_earl.csv",
            index=False
        )

        print("\nRESULTADO")

        print(
            "Recall:",
            f"{metricas['Recall']:.4f}"
        )

        print(
            "F1:",
            f"{metricas['F1']:.4f}"
        )

        print(
            "FPR:",
            f"{metricas['FPR']:.4f}"
        )

        print(
            "Falsos negativos:",
            metricas["FN"]
        )

        del modelo
        gc.collect()


# ==================================================
# 14. RESUMO ENTRE AS CINCO SEMENTES
# ==================================================

df = pd.DataFrame(resultados)

resumo = df.groupby("Modelo").agg(
    Recall_medio=("Recall", "mean"),
    Recall_std=("Recall", "std"),
    Precisao_media=("Precisao", "mean"),
    F1_medio=("F1", "mean"),
    F1_std=("F1", "std"),
    FPR_media=("FPR", "mean"),
    FPR_std=("FPR", "std"),
    FPR_max=("FPR", "max"),
    FP_medio=("FP", "mean"),
    FN_medio=("FN", "mean"),
    Parametros=("Parametros", "first"),
    Epocas_medias=("Epocas", "mean"),
    N_sementes=("Semente", "nunique")
).reset_index()

if not (
    resumo["N_sementes"] == len(SEMENTES)
).all():
    raise ValueError(
        "Faltam execucoes de alguma configuracao."
    )

resumo["FPR_aceitavel"] = (
    resumo["FPR_max"] <= FPR_MAXIMA
)

resumo.to_csv(
    SAIDA / "resumo_earl.csv",
    index=False
)


# ==================================================
# 15. RESUMO POR CLASSE
# ==================================================

df_classes = pd.DataFrame(
    classes_resultados
)

resumo_classes = df_classes.groupby(
    ["Modelo", "Classe"]
).agg(
    Recall_medio=("Recall_classe", "mean"),
    Recall_std=("Recall_classe", "std"),
    FN_medio=("FN", "mean"),
    Total=("Total", "first")
).reset_index()

resumo_classes.to_csv(
    SAIDA / "resumo_classes_earl.csv",
    index=False
)


# ==================================================
# 16. COMPARACAO PAREADA COM BASE
# ==================================================

base = df[
    df["Modelo"] == "AEC-REF"
][[
    "Semente",
    "Recall",
    "F1",
    "FPR",
    "FN"
]].copy()

base = base.rename(
    columns={
        "Recall": "Recall_AEC_REF",
        "F1": "F1_AEC_REF",
        "FPR": "FPR_AEC_REF",
        "FN": "FN_AEC_REF"
    }
)

comparacao = df[
    df["Modelo"] != "AEC-REF"
].merge(
    base,
    on="Semente",
    how="left",
    validate="many_to_one"
)

comparacao["Delta_Recall_pp"] = (
    100 * (
        comparacao["Recall"]
        - comparacao["Recall_AEC_REF"]
    )
)

comparacao["Delta_F1_pp"] = (
    100 * (
        comparacao["F1"]
        - comparacao["F1_AEC_REF"]
    )
)

comparacao["Delta_FPR_pp"] = (
    100 * (
        comparacao["FPR"]
        - comparacao["FPR_AEC_REF"]
    )
)

comparacao["Delta_FN"] = (
    comparacao["FN"]
    - comparacao["FN_AEC_REF"]
)

comparacao.to_csv(
    SAIDA / "comparacao_pareada_earl.csv",
    index=False
)

resumo_pareado = comparacao.groupby(
    "Modelo"
).agg(
    Delta_Recall_medio_pp=(
        "Delta_Recall_pp", "mean"
    ),
    Delta_F1_medio_pp=(
        "Delta_F1_pp", "mean"
    ),
    Delta_FPR_medio_pp=(
        "Delta_FPR_pp", "mean"
    ),
    Delta_FN_medio=(
        "Delta_FN", "mean"
    ),
    Sementes_melhores=(
        "Delta_Recall_pp",
        lambda valores: int(
            (valores > 0).sum()
        )
    )
).reset_index()

resumo_pareado.to_csv(
    SAIDA / "resumo_pareado_earl.csv",
    index=False
)


# ==================================================
# 17. SELECAO POR VALIDACAO
# ==================================================

# Criterio do experimento:
# 1. FPR maxima <= 10.5%
# 2. Maior F1 medio
# 3. Maior Recall medio
# 4. Menor numero de parametros

candidatos = resumo[
    resumo["FPR_aceitavel"]
].copy()

if candidatos.empty:

    print(
        "\nNenhum modelo respeitou a FPR maxima."
    )

else:

    candidatos = candidatos.sort_values(
        by=[
            "F1_medio",
            "Recall_medio",
            "Parametros"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )

    melhor = candidatos.iloc[0]

    nome_melhor = melhor["Modelo"]

    # Escolher a semente apenas com validacao.
    execucoes = df[
        df["Modelo"] == nome_melhor
    ].sort_values(
        by=["F1", "Recall"],
        ascending=[False, False]
    )

    execucao = execucoes.iloc[0]

    selecionado = {
        "Modelo": nome_melhor,
        "Semente": int(
            execucao["Semente"]
        ),
        "Percentil": PERCENTIL,
        "Limiar": float(
            execucao["Limiar"]
        ),
        "Lambda": float(
            execucao["Lambda"]
        ),
        "Beta": float(
            execucao["Beta"]
        ),
        "Recall_medio": float(
            melhor["Recall_medio"]
        ),
        "Recall_std": float(
            melhor["Recall_std"]
        ),
        "F1_medio": float(
            melhor["F1_medio"]
        ),
        "FPR_max": float(
            melhor["FPR_max"]
        ),
        "Parametros": int(
            melhor["Parametros"]
        )
    }

    with open(
        SAIDA / "modelo_selecionado_earl.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            selecionado,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nMODELO SELECIONADO")
    print(selecionado)


# ==================================================
# 18. FINALIZACAO
# ==================================================

print("\n" + "=" * 65)
print("EARL FINALIZADO")
print("=" * 65)

print("\nRESUMO")

print(
    resumo[
        [
            "Modelo",
            "Recall_medio",
            "Recall_std",
            "F1_medio",
            "FPR_media",
            "FPR_max",
            "FN_medio",
            "FPR_aceitavel"
        ]
    ].to_string(index=False)
)

print("\nCOMPARACAO COM O BASE")

print(
    resumo_pareado.to_string(index=False)
)

print("\nArquivos salvos em:")
print(SAIDA)

print(
    "\nO conjunto de teste final "
    "nao foi utilizado."
)
