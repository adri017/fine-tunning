import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1) configuración general
MODELO_BASE = "distilroberta-base"
SEMILLA = 42
LONGITUD_MAXIMA = 128

TAM_GO_TRAIN = 1500
TAM_GO_VAL = 300
TAM_GO_TEST = 300

TAM_IMDB_TRAIN = 1600
TAM_IMDB_VAL = 400
TAM_IMDB_TEST = 400

os.makedirs("resultados", exist_ok=True)
os.makedirs("graficas", exist_ok=True)

tokenizador = AutoTokenizer.from_pretrained(MODELO_BASE)

# 2) función de tokenización
def tokenizar_textos(ejemplos):
    return tokenizador(
        ejemplos["text"],
        truncation=True,
        padding="max_length",
        max_length=LONGITUD_MAXIMA
    )

# 3) métricas
def calcular_metricas(eval_pred):
    logits, etiquetas_reales = eval_pred
    predicciones = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(etiquetas_reales, predicciones),
        "f1": f1_score(etiquetas_reales, predicciones, average="macro"),
        "precision": precision_score(etiquetas_reales, predicciones, average="macro", zero_division=0),
        "recall": recall_score(etiquetas_reales, predicciones, average="macro", zero_division=0),
    }

# 4) preparar goemotions
def preparar_goemotions():
    bruto = load_dataset("go_emotions", "simplified")

    nombres_etiquetas = bruto["train"].features["labels"].feature.names
    num_etiquetas = len(nombres_etiquetas)

    def tiene_una_sola_etiqueta(ejemplo):
        return len(ejemplo["labels"]) == 1

    def crear_etiqueta_simple(ejemplo):
        return {"label": ejemplo["labels"][0]}

    procesado = {}

    for nombre_split, limite in [
        ("train", TAM_GO_TRAIN),
        ("validation", TAM_GO_VAL),
        ("test", TAM_GO_TEST),
    ]:
        ds = bruto[nombre_split]
        ds = ds.filter(tiene_una_sola_etiqueta)
        ds = ds.map(crear_etiqueta_simple)
        ds = ds.shuffle(seed=SEMILLA)
        ds = ds.select(range(min(limite, len(ds))))
        ds = ds.map(tokenizar_textos, batched=True)

        columnas_a_eliminar = [
            col for col in ds.column_names
            if col not in ["input_ids", "attention_mask", "label"]
        ]
        ds = ds.remove_columns(columnas_a_eliminar)

        procesado[nombre_split] = ds

    return procesado["train"], procesado["validation"], procesado["test"], num_etiquetas

# 5) preparar imdb
def preparar_imdb():
    bruto = load_dataset("imdb")

    train_completo = bruto["train"].shuffle(seed=SEMILLA).select(
        range(min(TAM_IMDB_TRAIN + TAM_IMDB_VAL, len(bruto["train"])))
    )

    division = train_completo.train_test_split(test_size=TAM_IMDB_VAL, seed=SEMILLA)

    ds_train = division["train"]
    ds_val = division["test"]
    ds_test = bruto["test"].shuffle(seed=SEMILLA).select(
        range(min(TAM_IMDB_TEST, len(bruto["test"])))
    )

    ds_train = ds_train.map(tokenizar_textos, batched=True)
    ds_val = ds_val.map(tokenizar_textos, batched=True)
    ds_test = ds_test.map(tokenizar_textos, batched=True)

    columnas_eliminar_train = [
        col for col in ds_train.column_names
        if col not in ["input_ids", "attention_mask", "label"]
    ]
    columnas_eliminar_val = [
        col for col in ds_val.column_names
        if col not in ["input_ids", "attention_mask", "label"]
    ]
    columnas_eliminar_test = [
        col for col in ds_test.column_names
        if col not in ["input_ids", "attention_mask", "label"]
    ]

    ds_train = ds_train.remove_columns(columnas_eliminar_train)
    ds_val = ds_val.remove_columns(columnas_eliminar_val)
    ds_test = ds_test.remove_columns(columnas_eliminar_test)

    num_etiquetas = 2
    return ds_train, ds_val, ds_test, num_etiquetas

# 6) configuraciones de experimentos
EXPERIMENTOS = [
    {
        "nombre": "configuracion_1_base",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 2,
        "weight_decay": 0.01
    },
    {
        "nombre": "configuracion_2_lr_alto",
        "learning_rate": 5e-5,
        "batch_size": 16,
        "epochs": 2,
        "weight_decay": 0.01
    },
    {
        "nombre": "configuracion_3_batch_pequeno_mas_epochs",
        "learning_rate": 2e-5,
        "batch_size": 8,
        "epochs": 3,
        "weight_decay": 0.01
    }
]

# 7) función para entrenar y evaluar
def ejecutar_experimento(nombre_dataset, ds_train, ds_val, ds_test, num_etiquetas, config_exp):
    print(f"dataset: {nombre_dataset} | experimento: {config_exp['nombre']}")

    modelo = AutoModelForSequenceClassification.from_pretrained(
        MODELO_BASE,
        num_labels=num_etiquetas
    )

    directorio_salida = f"resultados/{nombre_dataset}_{config_exp['nombre']}"

    argumentos_entrenamiento = TrainingArguments(
        output_dir=directorio_salida,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config_exp["learning_rate"],
        per_device_train_batch_size=config_exp["batch_size"],
        per_device_eval_batch_size=config_exp["batch_size"],
        num_train_epochs=config_exp["epochs"],
        weight_decay=config_exp["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=SEMILLA
    )

    entrenador = Trainer(
        model=modelo,
        args=argumentos_entrenamiento,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=calcular_metricas
    )

    entrenador.train()

    # evaluación en validación
    resultados_val = entrenador.evaluate(eval_dataset=ds_val)

    # evaluación final en test
    resultados_test = entrenador.evaluate(eval_dataset=ds_test)

    # guardar gráfica de pérdidas
    perdidas_train = [x["loss"] for x in entrenador.state.log_history if "loss" in x]
    perdidas_val = [x["eval_loss"] for x in entrenador.state.log_history if "eval_loss" in x]

    if len(perdidas_train) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(perdidas_train, marker="o", label="pérdida entrenamiento")
        if len(perdidas_val) > 0:
            plt.plot(perdidas_val, marker="s", label="pérdida validación")
        plt.title(f"loss - {nombre_dataset} - {config_exp['nombre']}")
        plt.xlabel("registro")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"graficas/loss_{nombre_dataset}_{config_exp['nombre']}.png")
        plt.close()

    # matriz de confusión solo para datasets con pocas clases
    if num_etiquetas <= 10:
        salida_pred = entrenador.predict(ds_test)
        y_real = salida_pred.label_ids
        y_pred = np.argmax(salida_pred.predictions, axis=1)

        matriz = confusion_matrix(y_real, y_pred)
        visualizador = ConfusionMatrixDisplay(confusion_matrix=matriz)
        fig, ax = plt.subplots(figsize=(6, 6))
        visualizador.plot(ax=ax)
        plt.title(f"matriz de confusión - {nombre_dataset} - {config_exp['nombre']}")
        plt.tight_layout()
        plt.savefig(f"graficas/confusion_{nombre_dataset}_{config_exp['nombre']}.png")
        plt.close()

    fila_resultado = {
        "dataset": nombre_dataset,
        "experimento": config_exp["nombre"],
        "learning_rate": config_exp["learning_rate"],
        "batch_size": config_exp["batch_size"],
        "epochs": config_exp["epochs"],
        "weight_decay": config_exp["weight_decay"],
        "val_accuracy": resultados_val.get("eval_accuracy"),
        "val_f1": resultados_val.get("eval_f1"),
        "val_precision": resultados_val.get("eval_precision"),
        "val_recall": resultados_val.get("eval_recall"),
        "test_accuracy": resultados_test.get("eval_accuracy"),
        "test_f1": resultados_test.get("eval_f1"),
        "test_precision": resultados_test.get("eval_precision"),
        "test_recall": resultados_test.get("eval_recall")
    }

    return fila_resultado

# 8) carga de datasets
print("preparando goemotions...")
go_train, go_val, go_test, go_num_etiquetas = preparar_goemotions()

print("preparando imdb...")
imdb_train, imdb_val, imdb_test, imdb_num_etiquetas = preparar_imdb()

# 9) ejecutar todos los experimentos
todos_los_resultados = []

# experimentos con goemotions
for exp in EXPERIMENTOS:
    resultado = ejecutar_experimento(
        nombre_dataset="goemotions",
        ds_train=go_train,
        ds_val=go_val,
        ds_test=go_test,
        num_etiquetas=go_num_etiquetas,
        config_exp=exp
    )
    todos_los_resultados.append(resultado)

# experimentos con imdb
for exp in EXPERIMENTOS:
    resultado = ejecutar_experimento(
        nombre_dataset="imdb",
        ds_train=imdb_train,
        ds_val=imdb_val,
        ds_test=imdb_test,
        num_etiquetas=imdb_num_etiquetas,
        config_exp=exp
    )
    todos_los_resultados.append(resultado)

# 10) guardar resultados finales
tabla_resultados = pd.DataFrame(todos_los_resultados)
tabla_resultados.to_csv("resumen_resultados.csv", index=False)

print("\nresumen final:")
print(tabla_resultados)

print("\narchivo guardado en: resultados/resumen_resultados.csv")
print("gráficas guardadas en la carpeta: graficas")
