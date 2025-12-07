# 🤝 Guia de Contribuição

Obrigado por considerar contribuir para este projeto! Este documento fornece diretrizes para contribuir de forma efetiva.

## 📋 Código de Conduta

Este projeto adere a um código de conduta. Ao participar, você concorda em manter um ambiente respeitoso e colaborativo.

## 🚀 Como Contribuir

### Reportar Bugs

Ao reportar um bug, inclua:

- **Descrição clara** do problema
- **Passos para reproduzir**
- **Comportamento esperado** vs **comportamento atual**
- **Ambiente**: OS, Python version, dependências
- **Logs/Screenshots** se aplicável

**Template:**

```markdown
## Descrição

[Descrição clara e concisa do bug]

## Reprodução

1. Executar '...'
2. Usar input '...'
3. Observar erro '...'

## Comportamento Esperado

[O que deveria acontecer]

## Ambiente

- OS: [e.g., Windows 11]
- Python: [e.g., 3.9]
- Transformers: [e.g., 4.35.0]

## Logs
```

[cole os logs aqui]

```

```

### Sugerir Melhorias

Para sugestões de features:

- **Use caso claro**
- **Benefícios esperados**
- **Implementação proposta** (se tiver ideia)
- **Alternativas consideradas**

### Pull Requests

1. **Fork** o repositório
2. **Crie uma branch** para sua feature:
   ```bash
   git checkout -b feature/MinhaFeature
   ```
3. **Faça commits** descritivos:
   ```bash
   git commit -m "feat: adiciona suporte para multi-label classification"
   ```
4. **Adicione testes** se aplicável
5. **Atualize documentação**
6. **Push** para sua branch:
   ```bash
   git push origin feature/MinhaFeature
   ```
7. **Abra um Pull Request**

## 📝 Convenções de Código

### Python Style Guide

Seguimos [PEP 8](https://pep8.org/):

```python
# ✅ Bom
def tokenize_function(example: dict) -> dict:
    """
    Tokeniza o texto de entrada.

    Args:
        example: Dicionário com campo 'prompt'

    Returns:
        dict: Tokens e attention masks
    """
    return tokenizer(
        example["prompt"],
        truncation=True,
        padding=True
    )

# ❌ Ruim
def tokenize(ex):
    return tokenizer(ex["prompt"],truncation=True,padding=True)
```

### Commits

Seguimos [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: adiciona nova feature
fix: corrige bug específico
docs: atualiza documentação
style: formatação de código
refactor: refatoração sem mudança de funcionalidade
test: adiciona ou modifica testes
chore: tarefas de manutenção
```

**Exemplos:**

```bash
git commit -m "feat: adiciona suporte para 3+ classes"
git commit -m "fix: corrige erro de encoding em textos especiais"
git commit -m "docs: atualiza README com exemplos de uso"
```

### Docstrings

Use docstrings no estilo Google:

```python
def predict(text: str, threshold: float = 0.5) -> dict:
    """
    Realiza predição em um texto.

    Args:
        text: Texto de entrada para classificação
        threshold: Limite mínimo de confiança (0-1)

    Returns:
        dict: Contém 'class', 'confidence' e 'is_certain'

    Raises:
        ValueError: Se text estiver vazio

    Example:
        >>> predict("Preciso de ajuda")
        {'class': 'suporte', 'confidence': 0.95, 'is_certain': True}
    """
    if not text:
        raise ValueError("Text cannot be empty")

    result = classifier(text)[0]
    return {
        'class': result['label'],
        'confidence': result['score'],
        'is_certain': result['score'] >= threshold
    }
```

## 🧪 Testes

### Executar Testes

```bash
# Todos os testes
pytest

# Com cobertura
pytest --cov=. --cov-report=html

# Específico
pytest tests/test_inference.py
```

### Escrever Testes

```python
import pytest
from inference import predict

def test_predict_suporte():
    """Testa classificação de mensagem de suporte."""
    result = predict("Meu produto veio com defeito")
    assert result['class'] == 'suporte'
    assert result['confidence'] > 0.5

def test_predict_empty_text():
    """Testa erro com texto vazio."""
    with pytest.raises(ValueError):
        predict("")

@pytest.mark.parametrize("text,expected", [
    ("Quero comprar", "venda"),
    ("Como configurar?", "suporte"),
])
def test_predict_multiple(text, expected):
    """Testa múltiplos casos."""
    result = predict(text)
    assert result['class'] == expected
```

## 📁 Estrutura de Código

Organize contribuições assim:

```
project/
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   ├── models/
│   │   ├── bert_classifier.py
│   │   └── trainer.py
│   ├── inference/
│   │   └── predictor.py
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_inference.py
├── notebooks/
│   └── fine_tunning_bart.ipynb
└── docs/
    ├── README.md
    └── DOCUMENTATION.md
```

## 🔍 Code Review

Ao revisar PRs, verificamos:

- [ ] **Funcionalidade**: Código faz o que propõe?
- [ ] **Testes**: Tem testes adequados?
- [ ] **Documentação**: Código e docs atualizados?
- [ ] **Style**: Segue convenções do projeto?
- [ ] **Performance**: Não degrada performance?
- [ ] **Segurança**: Não introduz vulnerabilidades?

## 📊 Contribuindo com Dados

### Adicionar Dados de Treinamento

```json
// Formato correto
{"prompt": "Texto da mensagem do cliente", "completion": "suporte"}
{"prompt": "Outro exemplo de mensagem", "completion": "venda"}
```

**Diretrizes:**

- Textos reais e variados
- Balanceamento entre classes
- Sem informações sensíveis (PII)
- Validar qualidade antes de commit

### Data Quality Checklist

- [ ] Dados anonimizados
- [ ] Labels corretos
- [ ] Textos limpos (sem HTML, etc)
- [ ] Distribuição balanceada
- [ ] Arquivo no formato JSONL

## 🐛 Debugging

### Habilitar Logs Detalhados

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Seu código aqui
predict("texto de teste")

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 📚 Recursos

### Aprender Mais

- [Transformers Course](https://huggingface.co/course)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

### Ferramentas Úteis

- **Linting**: `flake8`, `pylint`
- **Formatting**: `black`, `isort`
- **Type Checking**: `mypy`
- **Testing**: `pytest`, `pytest-cov`

## ❓ Dúvidas?

- Abra uma [Discussion](https://github.com/seu-usuario/seu-repo/discussions)
- Entre em contato: seu-email@example.com

## 🎉 Reconhecimento

Contribuidores serão listados no README e em releases notes!

---

**Obrigado por contribuir! 🚀**
