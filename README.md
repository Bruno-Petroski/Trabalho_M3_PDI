## Alunos:
- Bernardo Escobar
- Bruno Petroski Enghi
-  Gabriel Bortoloci
-  Laís Blum

## Links:
Link Colab: https://colab.research.google.com/drive/1tDkqu9GQrdQlHLaYSYuYu7QefDosXjA7?usp=sharing  

Sem problemas. Aqui está o texto completo do relatório para você copiar:

***

# Relatório Técnico: Detecção de Caracteres em Placas Veiculares com YOLOv11

**Disciplina:** Trabalho M2
**Alunos:**
* Bernardo Escobar
* Bruno Petroski Enghi
* Gabriel Bortoloci
* Laís Blum

## 1. Introdução

O presente trabalho tem como objetivo o desenvolvimento de um sistema de visão computacional capaz de identificar e classificar caracteres alfanuméricos em placas veiculares. Para tal, utilizou-se o framework **Ultralytics YOLOv11**.

O foco principal deste relatório reside na metodologia de preparação dos dados, especificamente na estratégia de separação do dataset para garantir a integridade dos testes, e na configuração do treinamento do modelo supervisionado.

## 2. Metodologia

### 2.1. Dataset e Pré-processamento
O conjunto de dados (*dataset*) utilizado consiste em imagens de placas veiculares e seus respectivos arquivos de anotação (formato `.txt` padrão YOLO). O desafio central em datasets de placas é a presença de múltiplas imagens do mesmo veículo (mesma placa) em ângulos ou condições de iluminação diferentes.

Se fizéssemos uma divisão aleatória simples das imagens, correríamos o risco de **Data Leakage** (Vazamento de Dados), onde o modelo treinaria com a foto "A" de um carro e seria testado com a foto "B" do mesmo carro. Isso inflaria artificialmente as métricas de desempenho.

### 2.2. Estratégia de Agrupamento
Para solucionar o problema de vazamento de dados, implementou-se um algoritmo de **Agrupamento por Assinatura**.

A função `get_signature(txt_path)` foi desenvolvida para ler o arquivo de anotação e extrair a sequência de IDs das classes (os caracteres da placa). Esta sequência forma uma "assinatura única" (ex: `10-11-20-22...` correspondendo a `A-B-K-M...`).

O código agrupa os caminhos das imagens em um dicionário onde a chave é essa assinatura. Dessa forma, todas as fotos que contêm a mesma sequência de caracteres (mesma placa física) ficam, obrigatoriamente, no mesmo grupo.

Forma gerados 2074 grupos.

### 2.3. Divisão dos Dados (Split)
Após o agrupamento, a divisão não foi feita por imagem, mas sim por **grupos de placas**. As proporções definidas foram:

* **Treinamento (70%):** Utilizado para o aprendizado dos pesos da rede neural.
* **Validação (15%):** Utilizado para ajuste de hiperparâmetros durante o treino.
* **Teste (15%):** Dados inéditos, reservados exclusivamente para a avaliação final das métricas.

A utilização de `random.seed(42)` garante a reprodutibilidade do experimento, assegurando que a mistura dos grupos seja sempre a mesma em execuções futuras.

### 2.4. Definição das Classes e Configuração YAML
O código gera automaticamente o arquivo `data.yaml` necessário para o YOLO. Foram definidas **36 classes**, mapeadas como `char_0` a `char_35`, correspondendo aos dígitos (0-9) e letras (A-Z) do alfabeto padrão.

## 3. Arquitetura do Modelo e Treinamento

Optou-se pela utilização do modelo **YOLOv11s** (`yolo11s.pt`).

* **Versão Small (s):** Diferente da versão Nano (n), a versão Small possui uma rede ligeiramente mais profunda e com mais parâmetros. Isso proporciona uma maior capacidade de extração de características (melhor precisão), com um custo computacional aceitável para execução em GPU (Google Colab T4).

### 3.1. Hiperparâmetros de Treinamento
Os parâmetros definidos para o treinamento foram:

* **Epochs (10):** Número de passagens completas pelo dataset de treino.
* **Image Size (640):** Resolução de entrada da rede. 640x640 pixels é o padrão aceito pelo YOLO.
* **Batch Size (16):** Quantidade de imagens processadas simultaneamente antes da atualização dos gradientes.
* **Device (0):** Garante a utilização da GPU para aceleração de hardware.

## 4. Métricas de Avaliação

Ao final do treinamento, o script executa `model.val()` utilizando o conjunto de dados separado. As seguintes métricas são calculadas e exibidas:

1. **Mean Precision (MP):** Indica a exatidão das detecções positivas. Uma precisão alta significa poucos falsos positivos (o modelo não inventa caracteres onde não existem).
2. **Mean Recall (MR):** Indica a capacidade do modelo de encontrar todos os caracteres presentes. Um recall alto significa que o modelo deixa passar poucos caracteres.
3. **mAP50 (Mean Average Precision @ IoU 0.5):** A métrica padrão de ouro para detecção de objetos. Considera uma predição correta se a área de sobreposição (IoU) entre a caixa prevista e a real for maior que 50%.
4. **mAP50-95:** Uma métrica mais rigorosa que calcula a média do mAP em vários limiares de IoU (de 0.5 a 0.95).
5. **Mean F1-Score:** A média harmônica entre Precisão e Recall, fornecendo uma visão única do equilíbrio do modelo.

### 4.1. Resultados Obtidos
Após a execução do treinamento e validação no conjunto de teste separado, obtiveram-se os seguintes resultados quantitativos:

| Métrica | Valor Obtido | Descrição |
| :--- | :--- | :--- |
| **Mean Precision (MP)** | **0.9969** | Precisão extremamente alta, indicando quase zero falsos positivos. |
| **Mean Recall (MR)** | **0.9931** | Capacidade robusta de detecção, encontrando mais de 99% dos caracteres presentes. |
| **mAP@50** | **0.9942** | Desempenho quase perfeito considerando sobreposição de 50%. |
| **mAP@50-95** | **0.8862** | Alto rigor no ajuste das bounding boxes (caixas delimitadoras). |
| **Mean F1-Score** | **0.9948** | Equilíbrio excelente entre precisão e recall. |
