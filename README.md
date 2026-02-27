[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/gIUcegNI)

# Projeto 1: Regressão - Predição de Tarifas de Táxi (NYC)

Este repositório contém o desenvolvimento de um modelo de **Deep Learning** utilizando **PyTorch** para prever o valor total (`total_amount`) das corridas de táxi amarelo em Nova York, com base no dataset oficial da TLC de 2023.

## 📋 Descrição do Projeto
O objetivo principal é aplicar técnicas de regressão para estimar custos de viagens. O modelo processa 1 milhão de registros e utiliza engenharia de atributos para extrair informações temporais e espaciais cruciais para a precisão da predição.

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Framework Deep Learning:** PyTorch
* **Manipulação de Dados:** Pandas e Numpy
* **Visualização:** Matplotlib e Seaborn
* **Pré-processamento:** Scikit-Learn

## 🧠 Arquitetura do Modelo
Foi implementada uma Rede Neural Artificial (Multilayer Perceptron) com a seguinte estrutura:

1.  **Camada de Entrada:** 15 features (distância, duração, hora do dia, dia da semana, IDs de localização, etc).
2.  **Camadas Ocultas:**
    * Linear (15 -> 64) + Ativação ReLU
    * Linear (64 -> 32) + Ativação ReLU
3.  **Camada de Saída:** Linear (32 -> 1) para o valor escalar da tarifa.



## 🚀 Como Executar
1. Certifique-se de ter o arquivo `2023_Yellow_Taxi_Trip_Data_20260225.csv` no diretório raiz.
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
