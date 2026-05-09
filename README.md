````markdown
# ✍️ Classificação de Dígitos Manuscritos com MNIST - Machine Learning

Este projeto tem como objetivo reconhecer dígitos manuscritos utilizando técnicas de **Machine Learning** aplicadas ao famoso dataset **MNIST**.

O foco principal foi construir um fluxo completo de classificação, começando com um problema de **classificação binária** para identificar o dígito `5` e, posteriormente, evoluindo para uma abordagem de **classificação multiclasse**, capaz de reconhecer os dígitos de `0` a `9`.

---

# 🎯 Objetivo

Criar modelos de classificação capazes de identificar corretamente dígitos escritos à mão a partir de imagens em escala de cinza.

O projeto foi dividido em duas partes principais:

- classificação binária: identificar se uma imagem representa ou não o dígito `5`;
- classificação multiclasse: identificar corretamente qualquer dígito entre `0` e `9`.

Essa divisão permitiu consolidar conceitos fundamentais de avaliação de modelos de classificação antes de avançar para um problema mais completo.

---

# 📊 Base de Dados

Foi utilizado o dataset **MNIST**, uma das bases mais conhecidas para projetos de classificação de imagens.

A base contém:

- 70.000 imagens de dígitos manuscritos;
- imagens em escala de cinza;
- resolução de `28x28` pixels;
- 784 atributos por imagem;
- rótulos representando os dígitos de `0` a `9`.

Cada imagem é representada por um vetor com os valores dos pixels.

O dataset foi carregado utilizando:

```python
from sklearn.datasets import fetch_openml
```

---

# 🧾 Estrutura dos Dados

Cada imagem do MNIST possui dimensão de `28x28` pixels.

Para ser utilizada pelos modelos de Machine Learning, cada imagem foi transformada em um vetor com:

```text
28 x 28 = 784 atributos
```

Cada atributo representa a intensidade de um pixel da imagem.

A variável alvo representa o dígito correspondente à imagem, podendo assumir valores de:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

# 🔎 Etapas do Projeto

As principais etapas realizadas foram:

- carregamento do dataset;
- separação entre variáveis preditoras e variável alvo;
- visualização de imagens do dataset;
- separação entre conjunto de treino e teste;
- criação do problema de classificação binária;
- treinamento do modelo `SGDClassifier`;
- avaliação com validação cruzada;
- criação da matriz de confusão;
- cálculo de precision, recall e F1-score;
- análise da curva Precision/Recall;
- ajuste de threshold;
- comparação de modelos com ROC Curve e ROC AUC;
- treinamento com `RandomForestClassifier`;
- evolução para classificação multiclasse;
- padronização dos dados com `StandardScaler`;
- análise de erros com matriz de confusão normalizada.

---

# 🧹 Tratamento dos Dados

As imagens foram tratadas como vetores numéricos, onde cada valor representa a intensidade de um pixel.

Além disso, foi aplicada padronização dos dados utilizando:

```python
StandardScaler
```

A padronização foi utilizada principalmente para melhorar o desempenho de modelos sensíveis à escala dos dados, como o `SGDClassifier`.

---

# ⚙️ Padronização

A padronização transforma os dados para que tenham média próxima de `0` e desvio padrão próximo de `1`.

Essa etapa foi importante porque os valores dos pixels podem variar em diferentes escalas, e modelos baseados em gradiente tendem a se beneficiar de dados padronizados.

Exemplo da técnica utilizada:

```python
from sklearn.preprocessing import StandardScaler
```

---

# 🤖 Modelos Utilizados

Foram utilizados modelos de classificação como:

- `SGDClassifier`;
- `RandomForestClassifier`.

O `SGDClassifier` foi utilizado principalmente para a classificação binária e para compreender métricas fundamentais de classificação.

O `RandomForestClassifier` foi utilizado como modelo alternativo para comparação de desempenho.

---

# 🔢 Classificação Binária

Inicialmente, o problema foi transformado em uma tarefa de classificação binária.

A pergunta principal dessa etapa foi:

```text
O dígito é o número 5?
```

Dessa forma, os rótulos foram transformados em duas classes:

- `True`: quando o dígito era `5`;
- `False`: quando o dígito não era `5`.

Essa abordagem permitiu estudar com mais profundidade conceitos como:

- matriz de confusão;
- precisão;
- recall;
- F1-score;
- threshold;
- curva Precision/Recall;
- curva ROC;
- AUC.

---

# 📊 Validação Cruzada

Foi utilizada validação cruzada para avaliar o desempenho do modelo de forma mais robusta.

Essa técnica divide o conjunto de treino em diferentes partes, treinando e avaliando o modelo em múltiplas combinações.

Com isso, foi possível obter uma estimativa mais confiável do desempenho do classificador.

---

# 📌 Matriz de Confusão

A matriz de confusão foi utilizada para analisar os erros e acertos do modelo.

Ela permite observar:

- verdadeiros positivos;
- falsos positivos;
- verdadeiros negativos;
- falsos negativos.

Essa análise foi essencial para entender melhor o comportamento do modelo além da simples acurácia.

---

# 📏 Métricas de Avaliação

As principais métricas utilizadas foram:

- **Accuracy**;
- **Precision**;
- **Recall**;
- **F1-score**;
- **ROC Curve**;
- **ROC AUC**.

Também foram utilizadas técnicas de apoio, como:

- validação cruzada;
- matriz de confusão;
- análise da curva Precision/Recall;
- ajuste de threshold.

---

# 🎯 Precision, Recall e F1-score

Durante o projeto, foram analisadas métricas mais adequadas para problemas de classificação, especialmente em cenários onde a acurácia pode ser enganosa.

## Precision

Indica, entre todas as previsões positivas feitas pelo modelo, quantas realmente estavam corretas.

## Recall

Indica, entre todos os casos positivos reais, quantos o modelo conseguiu identificar corretamente.

## F1-score

Representa uma média harmônica entre precision e recall, sendo útil quando é necessário equilibrar as duas métricas.

---

# 📈 Curva Precision/Recall

A curva Precision/Recall foi utilizada para analisar a relação entre precisão e recall em diferentes thresholds.

Essa etapa ajudou a visualizar como o modelo se comporta ao priorizar:

- maior precisão;
- maior recall;
- equilíbrio entre as duas métricas.

---

# ⚖️ Ajuste de Threshold

Foi realizado o ajuste do limite de decisão, conhecido como `threshold`.

O threshold define a partir de qual pontuação o modelo classifica uma instância como positiva.

No contexto da classificação binária, o ajuste de threshold permitiu observar como o modelo muda seu comportamento ao tentar identificar o dígito `5`.

Essa análise foi importante para entender o trade-off entre:

- aumentar a precisão;
- aumentar o recall.

---

# 📉 ROC Curve e ROC AUC

A curva ROC foi utilizada para comparar o desempenho dos classificadores considerando diferentes thresholds.

Também foi calculada a métrica ROC AUC, que resume a capacidade do modelo de separar as classes positiva e negativa.

Essas métricas foram utilizadas para comparar modelos e analisar a qualidade geral da classificação.

---

# 🔢 Classificação Multiclasse

Após a classificação binária, o projeto evoluiu para uma tarefa de classificação multiclasse.

Nessa etapa, o objetivo passou a ser identificar corretamente qualquer dígito entre:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Essa parte do projeto permitiu compreender como algoritmos de classificação lidam com múltiplas classes.

---

# 🔍 Análise de Erros

Foi utilizada uma matriz de confusão normalizada para analisar os erros cometidos pelo modelo na classificação multiclasse.

Essa análise permitiu identificar quais dígitos eram mais confundidos entre si.

A análise de erros foi importante para entender limitações do modelo e possíveis caminhos de melhoria.

---

# 🧠 Conceitos Trabalhados

Este projeto ajudou a consolidar conceitos fundamentais de Machine Learning, como:

- classificação binária;
- classificação multiclasse;
- validação cruzada;
- matriz de confusão;
- precision;
- recall;
- F1-score;
- curva Precision/Recall;
- curva ROC;
- ROC AUC;
- ajuste de threshold;
- padronização dos dados;
- análise de erros;
- comparação entre modelos;
- avaliação de classificadores.

---

# 📌 Resultado

O projeto permitiu construir um fluxo completo de classificação utilizando Machine Learning, passando desde o carregamento das imagens até a análise detalhada das métricas e dos erros do modelo.

Mais importante do que apenas alcançar boas métricas, o principal objetivo foi consolidar conceitos essenciais de classificação e avaliação de modelos.

Foram trabalhadas etapas importantes como:

- visualização de imagens;
- transformação dos dados;
- criação de problema binário;
- avaliação com validação cruzada;
- análise de métricas além da acurácia;
- comparação entre modelos;
- classificação multiclasse;
- análise de erros com matriz de confusão.

Este projeto representa uma etapa importante no meu processo de aprendizado em **Ciência de Dados** e **Machine Learning**, principalmente por envolver imagens, classificação e avaliação mais detalhada de modelos.

---

# 🚀 Melhorias Futuras

Pretendo revisitar este projeto futuramente para:

- testar modelos mais avançados;
- aplicar técnicas de otimização de hiperparâmetros;
- explorar Deep Learning;
- utilizar Redes Neurais Convolucionais, como CNNs;
- melhorar a visualização das imagens;
- aprofundar a análise de erros;
- comparar diferentes algoritmos de classificação;
- avaliar o impacto de diferentes técnicas de pré-processamento;
- criar uma aplicação simples para testar imagens manualmente.

---

# 📚 Aprendizados

Este projeto foi importante para compreender que avaliar modelos de classificação vai muito além da acurácia.

Durante o desenvolvimento, foi possível entender melhor a importância de métricas como precision, recall e F1-score, principalmente quando existe diferença entre o custo de falsos positivos e falsos negativos.

Também foi possível compreender como o ajuste de threshold altera o comportamento do modelo e como a matriz de confusão pode ser usada para interpretar os erros de forma mais clara.

---


Mesmo existindo muitas possibilidades de melhoria, esta versão representa uma etapa importante na construção dos meus conhecimentos em Machine Learning.
````
