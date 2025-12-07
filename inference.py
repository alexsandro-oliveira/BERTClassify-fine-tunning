"""
Inference script para o modelo BERT Text Classifier.

Este script fornece funções para carregar o modelo treinado e realizar
predições em textos individuais ou em batch.

Example:
    $ python inference.py --text "Preciso de ajuda com meu pedido"
    $ python inference.py --file inputs.txt --output results.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BERTClassifier:
    """
    Wrapper para o modelo BERT de classificação de texto.
    
    Attributes:
        model_path: Caminho para o modelo treinado
        device: Dispositivo para inferência (cuda/cpu)
        classifier: Pipeline do Hugging Face
        label_map: Mapeamento de IDs para nomes de classes
    """
    
    def __init__(self, model_path: str = "./bert-validator-test", device: int = -1):
        """
        Inicializa o classificador.
        
        Args:
            model_path: Caminho para o diretório do modelo
            device: Device ID (-1 para CPU, 0+ para GPU)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.label_map = {0: "suporte", 1: "venda"}
        
        logger.info(f"Carregando modelo de {self.model_path}")
        self._load_model()
        logger.info("Modelo carregado com sucesso!")
    
    def _load_model(self):
        """Carrega o modelo e tokenizer."""
        try:
            self.classifier = pipeline(
                "text-classification",
                model=str(self.model_path),
                device=self.device
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Union[str, float]]:
        """
        Realiza predição em um único texto.
        
        Args:
            text: Texto para classificação
            return_confidence: Se True, retorna score de confiança
            
        Returns:
            dict: Contém 'class' e opcionalmente 'confidence'
            
        Example:
            >>> classifier = BERTClassifier()
            >>> result = classifier.predict("Preciso de ajuda")
            >>> print(result)
            {'class': 'suporte', 'confidence': 0.95}
        """
        if not text or not text.strip():
            logger.warning("Texto vazio recebido")
            return {'class': 'unknown', 'confidence': 0.0}
        
        try:
            result = self.classifier(text)[0]
            label_id = int(result['label'].split('_')[-1])
            
            output = {
                'class': self.label_map.get(label_id, 'unknown'),
                'text': text
            }
            
            if return_confidence:
                output['confidence'] = round(result['score'], 4)
            
            return output
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {'class': 'error', 'confidence': 0.0, 'error': str(e)}
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_confidence: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Realiza predição em múltiplos textos (batch processing).
        
        Args:
            texts: Lista de textos para classificação
            batch_size: Tamanho do batch para processamento
            return_confidence: Se True, retorna scores de confiança
            
        Returns:
            list: Lista de dicionários com resultados
            
        Example:
            >>> texts = ["Preciso de ajuda", "Quero comprar"]
            >>> results = classifier.predict_batch(texts)
        """
        if not texts:
            logger.warning("Lista de textos vazia")
            return []
        
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processando {len(texts)} textos em {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_results = self.classifier(batch)
                
                for text, result in zip(batch, batch_results):
                    label_id = int(result['label'].split('_')[-1])
                    
                    output = {
                        'class': self.label_map.get(label_id, 'unknown'),
                        'text': text
                    }
                    
                    if return_confidence:
                        output['confidence'] = round(result['score'], 4)
                    
                    results.append(output)
                    
            except Exception as e:
                logger.error(f"Erro no batch {i//batch_size + 1}: {e}")
                # Adicionar resultados de erro para este batch
                for text in batch:
                    results.append({
                        'class': 'error',
                        'text': text,
                        'confidence': 0.0,
                        'error': str(e)
                    })
        
        return results
    
    def predict_from_file(
        self,
        input_file: str,
        output_file: str = None,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Realiza predições a partir de um arquivo de entrada.
        
        Args:
            input_file: Arquivo com textos (um por linha)
            output_file: Arquivo para salvar resultados (JSON)
            batch_size: Tamanho do batch
            
        Returns:
            list: Lista de resultados
        """
        logger.info(f"Lendo textos de {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Lidos {len(texts)} textos")
        results = self.predict_batch(texts, batch_size=batch_size)
        
        if output_file:
            logger.info(f"Salvando resultados em {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results


def main():
    """Função principal para uso via CLI."""
    parser = argparse.ArgumentParser(
        description="BERT Text Classifier - Inference"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='./bert-validator-test',
        help='Caminho para o modelo treinado'
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='Texto único para classificação'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Arquivo com textos (um por linha)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Arquivo de saída para resultados (JSON)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Tamanho do batch para processamento'
    )
    
    parser.add_argument(
        '--device',
        type=int,
        default=-1,
        help='Device ID (-1 para CPU, 0+ para GPU)'
    )
    
    args = parser.parse_args()
    
    # Inicializar classificador
    classifier = BERTClassifier(model_path=args.model, device=args.device)
    
    # Processar entrada
    if args.text:
        # Texto único
        result = classifier.predict(args.text)
        print("\nResultado:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.file:
        # Arquivo de textos
        results = classifier.predict_from_file(
            args.file,
            output_file=args.output,
            batch_size=args.batch_size
        )
        
        print(f"\nProcessados {len(results)} textos")
        
        # Mostrar estatísticas
        classes = {}
        for r in results:
            classes[r['class']] = classes.get(r['class'], 0) + 1
        
        print("\nDistribuição de classes:")
        for cls, count in classes.items():
            print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
