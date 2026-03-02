from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import matplotlib.pyplot as plt


print("Cargando datasets...")

# 1. Cargamos los 3 datasets (Rutas ultra-estables)
# Voz del Personaje
dataset_emotions = load_dataset("go_emotions", "simplified")

# Validación de Lógica
dataset_logic = load_dataset("hellaswag")

# Generación de Lore (Usamos IMDB como fuente de estructuras narrativas)
dataset_lore = load_dataset("imdb")

# Configuración del Tokenizador
model_checkpoint = "distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Funciones de preprocesamiento específicas
def tokenize_emotions(examples):
    examples["label"] = [l[0] for l in examples["labels"]]
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    return tokenized

def tokenize_logic(examples):
    return tokenizer(examples["ctx"], truncation=True, padding="max_length", max_length=128)

def tokenize_lore(examples):
    # IMDB usa la columna 'text'
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Aplicar mapeo (Usamos solo una parte para ir rápido en el test)
print("Tokenizando datasets...")
print("Re-procesando etiquetas para clasificación simple...")
tokenized_emotions = dataset_emotions["train"].select(range(1000)).map(tokenize_emotions, batched=True)
# Es vital quitar las columnas originales que no son tensores
tokenized_emotions = tokenized_emotions.remove_columns(["text", "labels"])
tokenized_emotions.set_format("torch")

tokenized_logic = dataset_logic["train"].select(range(1000)).map(tokenize_logic, batched=True)
tokenized_lore = dataset_lore["train"].select(range(1000)).map(tokenize_lore, batched=True)

print("-" * 30)
print("¡Éxito! Los 3 datasets están listos para RPGHistoryMaker AI.")

# Cargamos las métricas 
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1": f1}

learning_rate = 2e-5
batch_size = 16
epochs = 3
weight_decay = 0.01

# Configuración del modelo
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=28 
)
# Argumentos de Entrenamiento
training_args = TrainingArguments(
    output_dir="./results_rpg_history",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    logging_steps=1,
    logging_dir='./logs',
    load_best_model_at_end=True,
    report_to="none"
)

# Inicializamos el Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_emotions,
    eval_dataset=tokenized_emotions,
    compute_metrics=compute_metrics,
)

print("Iniciando entrenamiento corregido...")
trainer.train()

# 1. Ejecutar Evaluación del primer modelo
print("\n--- Evaluando Configuración A (Voz del Personaje) ---")
results_A = trainer.evaluate()

# 2. Vamos a simular el cambio de un parámetro para la comparativa (Nivel 5)
# Cambiamos el learning_rate a uno más agresivo
new_training_args = TrainingArguments(
    output_dir="./results_config_B",
    learning_rate=5e-4,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    logging_steps=1,
    report_to="none"
)

trainer_B = Trainer(
    model=model, # Re-usamos el modelo
    args=new_training_args,
    train_dataset=tokenized_emotions,
    eval_dataset=tokenized_emotions,
    compute_metrics=compute_metrics,
)

print("\n--- Entrenando Configuración B (Learning Rate Alto) ---")
trainer_B.train()
results_B = trainer_B.evaluate()

# 3. Mostrar Tabla Comparativa (Para tu documentación)
print("\n" + "="*30)
print("TABLA COMPARATIVA DE RESULTADOS")
print("="*30)
print(f"Config A (LR 2e-5) -> Accuracy: {results_A['eval_accuracy']:.4f} | F1: {results_A['eval_f1']:.4f}")
print(f"Config B (LR 5e-4) -> Accuracy: {results_B['eval_accuracy']:.4f} | F1: {results_B['eval_f1']:.4f}")
print("="*30)

history_A = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
history_B = [log['loss'] for log in trainer_B.state.log_history if 'loss' in log]

print(f"Puntos de datos encontrados: Config A: {len(history_A)}, Config B: {len(history_B)}")

if len(history_A) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(history_A, label='Config A (Estable - 2e-5)', color='blue')
    plt.plot(history_B, label='Config B (Agresiva - 5e-4)', color='red')
    plt.title('RPGHistoryMaker AI: Comparativa de Pérdida (Loss)')
    plt.xlabel('Pasos de entrenamiento')
    plt.ylabel('Pérdida (Loss)')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparativa_final.png')
    print("¡Gráfica 'comparativa_final.png' generada con éxito!")
else:
    print("ERROR: No hay datos de pérdida. Asegúrate de poner logging_steps=1 en TrainingArguments.")