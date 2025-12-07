# 🤖 BERT Text Classifier - Suporte vs Vendas

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📋 Descrição

Modelo de classificação de texto baseado em BERT para categorizar automaticamente mensagens de clientes entre **Suporte** e **Vendas**. Útil para sistemas de triagem automática, chatbots e análise de tickets.

### 🎯 Casos de Uso

- Roteamento automático de tickets de atendimento
- Classificação de e-mails corporativos
- Triagem de mensagens em chatbots
- Análise de intenção do cliente

## 🏗️ Arquitetura

```
┌─────────────────┐
│  Texto de Input │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BERT Tokenizer │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BERT Model    │
│ (bert-base-     │
│  uncased)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Classification  │
│ Head (2 labels) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Suporte | Venda │
└─────────────────┘
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.8 ou superior
- GPU com CUDA (recomendado) ou CPU
- 4GB+ RAM

### Instalação

1. Clone o repositório:

```bash
git clone <seu-repositorio>
cd fine_tunning/models
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. (Opcional) Configure o Hugging Face:

```bash
huggingface-cli login
```

## 📊 Dataset

### Formato dos Dados

Os dados devem estar no formato JSONL com a seguinte estrutura:

```json
{"prompt": "Como faço para configurar o fogão elétrico?", "completion": "suporte"}
{"prompt": "Quero comprar um micro-ondas, vocês têm sugestões?", "completion": "venda"}
```

### Estrutura de Arquivos

```
├── train.jsonl          # Dados de treinamento
├── test.jsonl           # Dados de validação
└── data/                # (Opcional) Dados adicionais
```

### Classes

| Classe  | Label | Descrição                             |
| ------- | ----- | ------------------------------------- |
| suporte | 0     | Questões técnicas, problemas, ajuda   |
| venda   | 1     | Interesse em compra, preços, produtos |

## 🛠️ Treinamento

### Usando o Notebook

Execute as células do notebook `fine_tunning_bart.ipynb` sequencialmente:

1. **Autenticação** (se necessário)
2. **Carregamento dos dados**
3. **Tokenização**
4. **Configuração do treinamento**
5. **Treinamento do modelo**
6. **Avaliação**
7. **Salvamento**

### Hiperparâmetros

```python
num_train_epochs = 3
per_device_train_batch_size = 8
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 100
```

### Métricas de Avaliação

- **Accuracy**: Métrica principal
- **Loss**: Monitoramento do treinamento
- Avaliação a cada 200 steps

## 💻 Uso

### Inferência Básica

```python
from transformers import pipeline

# Carregar modelo
classifier = pipeline("text-classification",
                     model="./bert-validator-test")

# Fazer predição
texto = "Preciso de ajuda com meu pedido"
resultado = classifier(texto)
print(resultado)
# [{'label': 'LABEL_0', 'score': 0.95}]  # suporte
```

### Inferência com Labels Customizados

```python
label_names = {0: "suporte", 1: "venda"}

def classificar_texto(texto):
    resultado = classifier(texto)[0]
    label_id = int(resultado['label'].split('_')[-1])
    classe = label_names[label_id]
    confianca = resultado['score']

    return {
        'classe': classe,
        'confianca': confianca
    }

# Exemplo
resultado = classificar_texto("Quanto custa o produto X?")
print(f"Classe: {resultado['classe']}")
print(f"Confiança: {resultado['confianca']:.2%}")
```

### Batch Prediction

```python
textos = [
    "Meu produto veio com defeito",
    "Gostaria de comprar 5 unidades",
    "Como faço para resetar a senha?"
]

resultados = classifier(textos)
for texto, resultado in zip(textos, resultados):
    print(f"{texto} → {resultado['label']}")
```

## 📁 Estrutura do Projeto

```
fine_tunning/models/
│
├── fine_tunning_bart.ipynb    # Notebook principal
├── requirements.txt            # Dependências
├── README.md                   # Documentação
├── DOCUMENTATION.md            # Documentação técnica detalhada
│
├── train.jsonl                 # Dados de treinamento
├── test.jsonl                  # Dados de teste
│
├── bert-validator-test/        # Modelo treinado
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── checkpoint-*/           # Checkpoints intermediários
│
├── logs/                       # Logs de treinamento
└── data/                       # Dados adicionais
```

## 🔧 Configuração Avançada

### Ajuste de Hiperparâmetros

Para melhorar o desempenho, ajuste os parâmetros em `TrainingArguments`:

```python
training_args = TrainingArguments(
    output_dir="./bert-validator-test",
    num_train_epochs=5,              # Mais épocas
    learning_rate=3e-5,              # LR menor
    per_device_train_batch_size=16,  # Batch maior (se GPU permitir)
    warmup_ratio=0.1,                # 10% warmup
    weight_decay=0.01,
    fp16=True,                       # Mixed precision
)
```

### Data Augmentation

Para datasets pequenos, considere:

```python
from nlpaug.augmenter.word import SynonymAug

aug = SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(original_text)
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    # ... outros parâmetros
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

## 📈 Performance

### Resultados Esperados

Com o dataset padrão:

- **Accuracy**: ~85-95% (depende do dataset)
- **Training time**: ~10-30 min (GPU) / 1-3h (CPU)
- **Inference**: ~50-100ms por texto (GPU) / 200-500ms (CPU)

### Otimização de Inferência

Para produção, considere:

1. **ONNX Runtime**: 2-3x mais rápido
2. **Quantização**: Modelo menor, inferência mais rápida
3. **Batch processing**: Processar múltiplos textos juntos

## 🐛 Troubleshooting

### Erro de Memória (OOM)

Reduza o batch size:

```python
per_device_train_batch_size = 4  # ou menor
gradient_accumulation_steps = 8  # compensar batch menor
```

### Overfitting

- Aumentar `weight_decay`
- Adicionar dropout
- Usar data augmentation
- Coletar mais dados

### Underfitting

- Aumentar `num_train_epochs`
- Ajustar `learning_rate`
- Verificar qualidade dos dados

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👥 Autores

- Seu Nome - [GitHub](https://github.com/seu-usuario)

## 🙏 Agradecimentos

- [Hugging Face](https://huggingface.co/) pela biblioteca Transformers
- [BERT](https://arxiv.org/abs/1810.04805) - Devlin et al., 2018
- Comunidade open source

## 📚 Referências

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Fine-tuning Best Practices](https://huggingface.co/docs/transformers/training)

## 📞 Suporte

Para questões e suporte:

- Abra uma [issue](https://github.com/seu-usuario/seu-repo/issues)
- Email: seu-email@example.com

---

⭐ Se este projeto foi útil, considere dar uma estrela!
