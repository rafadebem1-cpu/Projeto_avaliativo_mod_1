
## Desenvolvimento de IA para Análise Preditiva

Projeto desenvolvido para a Situação de Aprendizagem (Projeto Avaliativo) do Módulo 1, com o objetivo de construir um pipeline completo de Ciência de Dados para prever falhas mecânicas em equipamentos industriais utilizando técnicas de Machine Learning.

---

# 📌 Objetivo

Desenvolver uma solução de Inteligência Artificial capaz de prever falhas em máquinas industriais a partir de dados coletados por sensores.

O projeto contempla todas as etapas de um pipeline de Ciência de Dados, desde a análise exploratória dos dados até o treinamento, avaliação e comparação de modelos preditivos.

---

# 🎯 Problema de Negócio

Em ambientes industriais, falhas inesperadas em equipamentos podem causar interrupções na produção, aumento dos custos de manutenção e perda de produtividade.

Este projeto busca prever antecipadamente a ocorrência de falhas mecânicas utilizando algoritmos de classificação supervisionada, permitindo que a empresa realize manutenções preventivas.

Variável alvo:

* **Falha = 1** → Equipamento com falha
* **Falha = 0** → Funcionamento normal

---

# 🧠 Técnicas Utilizadas

Durante o desenvolvimento foram aplicadas técnicas de:

* Análise Exploratória de Dados (EDA)
* Limpeza e tratamento dos dados
* Engenharia de Atributos (Feature Engineering)
* Balanceamento de classes
* Escalonamento de variáveis
* Ajuste de hiperparâmetros
* Controle de Overfitting
* Avaliação por Acurácia
* Comparação entre modelos de Machine Learning

---

# 🤖 Modelos de Machine Learning

Foram implementados e comparados os seguintes algoritmos:

* K-Nearest Neighbors (KNN)
* Árvore de Decisão (Decision Tree)

O modelo com melhor desempenho na base de teste será recomendado como solução final.

---

# 📊 Pipeline do Projeto

1. Importação da base de dados
2. Análise exploratória (EDA)
3. Limpeza e tratamento dos dados
4. Engenharia de atributos
5. Divisão treino e teste
6. Balanceamento dos dados (SMOTE ou Random Under Sampling)
7. Escalonamento das variáveis (StandardScaler)
8. Treinamento dos modelos
9. Ajuste de hiperparâmetros
10. Avaliação da acurácia
11. Comparação dos modelos
12. Conclusão

---

# 📁 Estrutura do Projeto

```text
Projeto/
│
├── dataset/
│   └── manutencao_preditiva.csv
│
├── projeto_avaliativo.ipynb
├── requirements.txt
├── README.md
└── anotacoes do Departamento de Engenharia.docx
```

---

# 🛠 Tecnologias Utilizadas

* Python 3
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Git
* GitHub

---

# 📈 Visualizações

O projeto utiliza gráficos para auxiliar a análise dos dados, incluindo:

* Histograma
* Gráfico de Barras
* Heatmap de Correlação
* Boxplots

---

# ⚙️ Como Executar

## 1. Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
```

## 2. Entrar na pasta

```bash
cd nome-do-projeto
```

## 3. Criar ambiente virtual

Windows

```bash
python -m venv .venv
```

## 4. Ativar o ambiente

```bash
.venv\Scripts\activate
```

## 5. Instalar as dependências

```bash
pip install -r requirements.txt
```

## 6. Executar o Notebook

Abra o arquivo `projeto_avaliativo.ipynb` no VS Code ou Jupyter Notebook e execute todas as células na ordem.

---

# 🌿 Organização do Git

O projeto utiliza Git para controle de versões.

Estratégia de branches:

* **main** → versão estável
* **develop** → integração das funcionalidades

Exemplos:

```
feature/eda
feature/data-prep
feature/feature-engineering
feature/knn
feature/decision-tree
```

---

# 📋 Requisitos Desenvolvidos

* ✔ Análise Exploratória (EDA)
* ✔ Limpeza dos Dados
* ✔ Feature Engineering
* ✔ Divisão Treino/Teste
* ✔ Balanceamento dos Dados
* ✔ StandardScaler
* ✔ Ajuste de Hiperparâmetros
* ✔ Controle de Overfitting
* ✔ Avaliação da Acurácia
* ✔ Comparação entre Modelos

---

# 🚀 Melhorias Futuras

Algumas melhorias que poderão ser implementadas:

* Interface Web utilizando Streamlit
* Dashboard interativo
* Validação Cruzada (Cross Validation)
* GridSearchCV para otimização automática
* Novos algoritmos de Machine Learning
* Exportação automática dos resultados
* Deploy da aplicação em nuvem

---

# 👥 Equipe

* Rafael de Oliveira Pinto

---

# 🎥 Vídeo de Apresentação

Link:

> (Inserir link do Google Drive)

---

# 📚 Referências

* Documentação oficial do Scikit-Learn
* Documentação do Pandas
* Documentação do NumPy
* Documentação do Seaborn
* Material didático da disciplina



