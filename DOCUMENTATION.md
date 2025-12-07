# 📖 Documentação Técnica - BERT Text Classifier

## Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Modelo](#arquitetura-do-modelo)
3. [Pipeline de Dados](#pipeline-de-dados)
4. [Processo de Treinamento](#processo-de-treinamento)
5. [Avaliação e Métricas](#avaliação-e-métricas)
6. [API de Inferência](#api-de-inferência)
7. [Deployment](#deployment)
8. [Monitoramento](#monitoramento)
9. [Manutenção](#manutenção)

---

## Visão Geral

### Objetivo

Desenvolver um classificador de texto binário capaz de distinguir entre mensagens de **suporte** e **vendas** com alta precisão e baixa latência.

### Tecnologias Utilizadas

| Componente      | Tecnologia               | Versão |
| --------------- | ------------------------ | ------ |
| Base Model      | BERT (bert-base-uncased) | -      |
| Framework       | Transformers             | 4.x+   |
| Deep Learning   | PyTorch                  | 2.x+   |
| Data Processing | Datasets                 | 2.x+   |
| Metrics         | Evaluate                 | 0.4+   |

### Requisitos de Sistema

**Mínimo:**

- CPU: 4 cores
- RAM: 8GB
- Storage: 2GB

**Recomendado:**

- GPU: NVIDIA com 8GB+ VRAM
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 5GB

---

## Arquitetura do Modelo

### Base Model: BERT

**BERT (Bidirectional Encoder Representations from Transformers)** é um modelo de linguagem pré-treinado que compreende contexto bidirecional.

#### Especificações

```
Model: bert-base-uncased
Parameters: 110M
Layers: 12
Hidden Size: 768
Attention Heads: 12
Max Sequence Length: 512 tokens
Vocabulary Size: 30,522
```

#### Arquitetura da Camada de Classificação

```python
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(...)
    (pooler): BertPooler(...)
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=2)
)
```

### Transfer Learning Strategy

1. **Frozen Embeddings**: Não (todas as camadas treináveis)
2. **Learning Rate**: 5e-5 (padrão para BERT fine-tuning)
3. **Warmup**: 100 steps para estabilidade inicial
4. **Weight Decay**: 0.01 para regularização

---

## Pipeline de Dados

### 1. Carregamento

```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "train.jsonl",
        "validation": "test.jsonl"
    }
)
```

**Formato JSONL:**

```json
{ "prompt": "texto do usuário", "completion": "suporte|venda" }
```

### 2. Pré-processamento

#### Transformação de Labels

```python
mapDict = {
    "suporte": 0,
    "venda": 1
}

def transform_labels(example):
    completion_value = example["completion"]
    return {"labels": mapDict[completion_value]}
```

#### Tokenização

```python
def tokenize_function(example):
    return tokenizer(
        example["prompt"],
        truncation=True,      # Limita a 512 tokens
        padding=True,         # Padeia para o batch
        max_length=512
    )
```

**Output do Tokenizer:**

- `input_ids`: IDs dos tokens
- `attention_mask`: Máscara de atenção (1 para tokens reais, 0 para padding)
- `token_type_ids`: Tipo de segmento (não usado aqui)

### 3. Data Collation

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

**Função:**

- Agrupa exemplos em batches
- Aplica padding dinâmico
- Otimiza uso de memória

---

## Processo de Treinamento

### Configuração do Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./bert-validator-test",
    overwrite_output_dir=True,

    # Treinamento
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,    # Effective batch size = 32

    # Otimizador
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,

    # Avaliação
    eval_strategy="steps",
    eval_steps=200,

    # Checkpoints
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,               # Mantém apenas 3 checkpoints

    # Best Model
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    # Logging
    logging_dir="./logs",
    logging_steps=100,

    # Performance
    fp16=True,                        # Mixed precision (requer GPU)
    dataloader_num_workers=2,
    seed=42,                          # Reprodutibilidade
)
```

### Componentes do Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model=model,                      # BertForSequenceClassification
    args=training_args,               # TrainingArguments
    train_dataset=train_dataset,      # Dataset tokenizado
    eval_dataset=eval_dataset,        # Dataset de validação
    data_collator=data_collator,      # DataCollatorWithPadding
    tokenizer=tokenizer,              # AutoTokenizer
    compute_metrics=compute_metrics   # Função de métricas
)
```

### Loop de Treinamento

**Por Época:**

```
For each epoch (1 to 3):
    For each batch in train_dataset:
        1. Forward pass
        2. Compute loss
        3. Backward pass (gradient computation)
        4. Gradient accumulation (if enabled)
        5. Optimizer step
        6. Learning rate scheduling

        If step % eval_steps == 0:
            - Evaluate on validation set
            - Log metrics
            - Save checkpoint (if best)
```

### Otimização

#### AdamW Optimizer

```python
# Configuração padrão (automática)
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

#### Learning Rate Schedule

```python
# Linear warmup seguido de decay linear
total_steps = len(train_dataset) * num_epochs
warmup_steps = 100

LR Schedule:
    steps 0-100: warmup (0 → 5e-5)
    steps 100-end: linear decay (5e-5 → 0)
```

---

## Avaliação e Métricas

### Métricas Implementadas

#### 1. Accuracy

```python
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

**Fórmula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

#### 2. Loss Function

```python
# Cross Entropy Loss (automático no BERT)
loss = CrossEntropyLoss()(logits, labels)
```

### Métricas Adicionais Recomendadas

```python
from sklearn.metrics import classification_report, confusion_matrix

def compute_extended_metrics(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }
```

### Interpretação dos Resultados

| Métrica   | Bom  | Excelente | Observações                    |
| --------- | ---- | --------- | ------------------------------ |
| Accuracy  | >80% | >90%      | Métrica principal              |
| Precision | >85% | >95%      | Poucos falsos positivos        |
| Recall    | >85% | >95%      | Poucos falsos negativos        |
| F1-Score  | >85% | >95%      | Balanço entre precision/recall |

---

## API de Inferência

### Carregamento do Modelo

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)

# Método 1: Pipeline (recomendado para produção)
classifier = pipeline(
    "text-classification",
    model="./bert-validator-test",
    device=0  # GPU (use -1 para CPU)
)

# Método 2: Manual
model = AutoModelForSequenceClassification.from_pretrained(
    "./bert-validator-test"
)
tokenizer = AutoTokenizer.from_pretrained(
    "./bert-validator-test"
)
```

### Inferência Single

```python
def predict(text: str) -> dict:
    """
    Realiza predição em um único texto.

    Args:
        text: Texto de entrada

    Returns:
        dict: {
            'label': str,
            'score': float,
            'class_name': str
        }
    """
    result = classifier(text)[0]
    label_id = int(result['label'].split('_')[-1])

    label_map = {0: "suporte", 1: "venda"}

    return {
        'label': result['label'],
        'score': result['score'],
        'class_name': label_map[label_id]
    }
```

### Inferência Batch

```python
def predict_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Realiza predição em múltiplos textos.

    Args:
        texts: Lista de textos
        batch_size: Tamanho do batch

    Returns:
        list[dict]: Lista de predições
    """
    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_results = classifier(batch)
        results.extend(batch_results)

    return results
```

### Performance de Inferência

| Configuração   | Throughput         | Latência      |
| -------------- | ------------------ | ------------- |
| CPU (single)   | ~2-5 texts/sec     | 200-500ms     |
| CPU (batch 32) | ~10-20 texts/sec   | 50-100ms/text |
| GPU (single)   | ~20-50 texts/sec   | 20-50ms       |
| GPU (batch 32) | ~100-200 texts/sec | 5-10ms/text   |

---

## Deployment

### 1. Containerização com Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar modelo e código
COPY bert-validator-test/ ./bert-validator-test/
COPY inference.py .

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["python", "inference.py"]
```

### 2. API REST com FastAPI

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="BERT Classifier API")

# Carregar modelo
classifier = pipeline(
    "text-classification",
    model="./bert-validator-test"
)

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    class_name: str
    confidence: float

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TextInput):
    try:
        result = classifier(input_data.text)[0]
        label_id = int(result['label'].split('_')[-1])
        label_map = {0: "suporte", 1: "venda"}

        return PredictionOutput(
            class_name=label_map[label_id],
            confidence=result['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### 3. Deployment na Cloud

#### AWS SageMaker

```python
from sagemaker.huggingface import HuggingFaceModel

huggingface_model = HuggingFaceModel(
    model_data="s3://bucket/model.tar.gz",
    role=role,
    transformers_version="4.26",
    pytorch_version="2.0",
    py_version="py39",
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)
```

#### Google Cloud Run

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: bert-classifier
spec:
  template:
    spec:
      containers:
        - image: gcr.io/project/bert-classifier:latest
          resources:
            limits:
              memory: 4Gi
              cpu: 2
```

---

## Monitoramento

### Métricas de Produção

```python
from prometheus_client import Counter, Histogram

# Contadores
predictions_total = Counter(
    'predictions_total',
    'Total predictions made',
    ['class']
)

# Histogramas
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)

# Uso
@prediction_latency.time()
def predict_with_monitoring(text):
    result = classifier(text)[0]
    label = result['label']
    predictions_total.labels(class=label).inc()
    return result
```

### Logging Estruturado

```python
import logging
import json

logger = logging.getLogger(__name__)

def log_prediction(text, result, latency):
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'input_length': len(text),
        'predicted_class': result['label'],
        'confidence': result['score'],
        'latency_ms': latency * 1000
    }
    logger.info(json.dumps(log_data))
```

### Alertas

Configure alertas para:

- **Latência alta**: >500ms (p95)
- **Confiança baixa**: <0.7 em >20% das predições
- **Taxa de erro**: >1%
- **Uso de memória**: >80%

---

## Manutenção

### Re-treinamento

**Quando re-treinar:**

- A cada 3-6 meses
- Quando accuracy cair >5%
- Ao adicionar novas categorias
- Com dataset expandido (>20% novo)

**Processo:**

```python
# 1. Backup do modelo atual
shutil.copytree("bert-validator-test", "bert-validator-backup")

# 2. Carregar modelo anterior como base
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-validator-backup"
)

# 3. Treinar com novos dados
trainer = Trainer(...)
trainer.train(resume_from_checkpoint=True)

# 4. Avaliar e comparar
old_metrics = evaluate_model("bert-validator-backup")
new_metrics = evaluate_model("bert-validator-test")

# 5. Deploy se melhorias > 2%
if new_metrics['accuracy'] > old_metrics['accuracy'] + 0.02:
    deploy_new_model()
```

### Versionamento

```python
# model_versions.json
{
    "v1.0.0": {
        "date": "2025-01-15",
        "accuracy": 0.89,
        "dataset_size": 1000,
        "path": "models/v1.0.0/"
    },
    "v1.1.0": {
        "date": "2025-04-20",
        "accuracy": 0.92,
        "dataset_size": 1500,
        "path": "models/v1.1.0/"
    }
}
```

### Debugging

#### Predições Incorretas

```python
def analyze_misclassifications(dataset, model):
    errors = []

    for example in dataset:
        pred = model(example['text'])
        if pred['label'] != example['true_label']:
            errors.append({
                'text': example['text'],
                'predicted': pred['label'],
                'actual': example['true_label'],
                'confidence': pred['score']
            })

    return errors
```

#### Data Drift Detection

```python
from scipy.stats import ks_2samp

def detect_drift(train_embeddings, prod_embeddings):
    statistic, p_value = ks_2samp(train_embeddings, prod_embeddings)

    if p_value < 0.05:
        return "DRIFT DETECTED"
    return "NO DRIFT"
```

---

## Referências Técnicas

### Papers

1. **BERT**: Devlin et al. (2018) - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
2. **Fine-tuning**: Howard & Ruder (2018) - [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)

### Documentação

- [Transformers Docs](https://huggingface.co/docs/transformers/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Datasets Docs](https://huggingface.co/docs/datasets/)

---

**Última atualização**: Dezembro 2025  
**Versão da documentação**: 1.0.0
