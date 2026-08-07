# Pós-processamento em CBIR

Experimento de recuperação de regiões em imagens de documentos. O sistema cria
regiões candidatas, extrai descritores visuais e ranqueia resultados combinando
similaridade de aparência e posição espacial.

CBIR significa *Content-Based Image Retrieval*: recuperação de imagens baseada
no próprio conteúdo visual, em vez de metadados ou nomes de arquivos.

## Pipeline

1. Geração de propostas de região com Selective Search.
2. Extração de pontos-chave e descritores ORB.
3. Comparação de descritores com Brute-Force Matcher e distância de Hamming.
4. Combinação da similaridade visual com a sobreposição espacial (IoU).
5. Remoção de resultados redundantes com Non-Maximum Suppression (NMS).
6. Ranking das regiões e visualização do melhor resultado.

## Tecnologias

- Python
- OpenCV e módulo `ximgproc`
- NumPy
- Matplotlib

## Como executar

### Pré-requisitos

- Python 3.10 ou superior
- Conjunto de imagens de documentos e consultas

```bash
git clone https://github.com/guilhermelerner/Pos-ProcessamentoCBIR.git
cd Pos-ProcessamentoCBIR
python -m venv .venv
source .venv/bin/activate
pip install opencv-contrib-python numpy matplotlib
```

No Windows, ative o ambiente com `.venv\\Scripts\\activate`.

Antes de executar, ajuste em `RecuperacaodeImagens.py` os caminhos de
`doc_files` e `query_files` para as pastas do seu conjunto de imagens:

```bash
python RecuperacaodeImagens.py
```

## Parâmetros úteis

- `max_proposals`: limita a quantidade de regiões candidatas por documento.
- `alpha`: controla o peso entre similaridade visual e IoU.
- `limite_iou`: define quando resultados sobrepostos são considerados duplicados.

## Resultados

O repositório inclui exemplos visuais de consultas processadas e um relatório
com o contexto, a metodologia e a avaliação do experimento.

## Limitações

Os caminhos do conjunto de dados ainda são locais e precisam ser configurados.
ORB é uma abordagem clássica e eficiente, mas pode perder robustez em imagens
com pouco contraste, grandes transformações ou poucos pontos-chave.
