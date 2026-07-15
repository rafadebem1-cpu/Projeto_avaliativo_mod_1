# importar as bibliotecas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.metrics import accuracy_score # knn
from sklearn.tree import DecisionTreeClassifier # arvore de decisao

from imblearn.over_sampling import SMOTE




# abrir o dataframe e atribuir na variavel "df"
df = pd.read_csv("manutencao_preditiva.csv")


df.head()

# primeira visao do dataframe


df. shape  

# mostra o tamanho do dataframe, ou seja, 10.000 linhas e 14 colunas

df.info()

# Mostra resumo das informações gerais: colunas, tipos de dados e valores nulos (comando "df.info")
# no caso do dataframe do projeto, ele tem 14 colunas e 10.000 linhas (comando  "df.shape")
# sendo que 4 colunas possuem dados nulos 


df.isnull(). sum()

# Verifica valores vazios
# no caso tem 4 colunas com 500 linhas com elementos vazios em cada uma

df.describe()
# mostra Estatísticas das colunas numéricas (media, desvio padrao, mediana, min, max, quartis)

# apos uma visualzação inicial, optei por criar um novo dataframe 'df_tratado' e fiz uma copia do 'df' original 
# caso precise resgatar informacoes ou voltar para original

# DataFrame original
df

# Criar uma cópia para tratamento e chama-la de 'df_tratado'
df_tratado = df.copy()


## Remover colunas que não agregam informação
# como o projeto quer identificar falha essas duas colunas nao interferem
## 'udi' e 'id_produto'

df_tratado.drop(columns= ["udi"], inplace=True)
df_tratado.drop(columns= ["id_produto"], inplace=True)


# procurar dados duplicados

df_tratado.duplicated().sum()

# visualizar as linhas duplicadas

duplicados = df_tratado[df_tratado.duplicated()]

duplicados

# remover os duplicados

df_tratado = df_tratado.drop_duplicates()

# visualizacao apos dados duplicados removidos

df_tratado.duplicated().sum()

df_tratado.isnull(). sum()

# converter a coluna "tipo" de str para numerica

df_tratado = pd.get_dummies(
    df_tratado,
    columns=["tipo"],
    dtype=int
)

#visualizar df_tratado e conferir se apareceu a coluna "tipo" como numerica
df_tratado.info()


# revisao de dados nulos, pois apos apagar os dados duplicados, o dataframe ficom com 315 nulos

df_tratado.isnull().sum()

# considerando os dados nulos em 4 colunas (São ainda 315 valores nulos)
# preencher dados nulos com a mediana pois ela é menos sensivel aos valores extremos

for coluna in [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm"
]:
    df_tratado[coluna] = df_tratado[coluna].fillna(df[coluna].median())

# consulta dados nulos apos a inclusao da mediana

df_tratado.isnull(). sum()

df_tratado.head()

df_tratado.shape

# Histograma das variáveis preditoras
# analisar a distribuição dos dados, identificar assimetrias e possíveis outliers

variaveis = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min"
]

plt.figure(figsize=(14,8))

for i, coluna in enumerate(variaveis):
    plt.subplot(2,3,i+1)
    sns.histplot(df_tratado[coluna], bins=30, kde=True)
    plt.title(coluna)

plt.tight_layout()
plt.show()

# 2. Gráfico de barras da variável alvo
# Verifica o desbalanceamento da base

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

plt.figure(figsize=(7,5))

ax = sns.countplot(
    x="falha_maquina",
    data=df_tratado
)

plt.title("Distribuição da Variável Alvo")
plt.xlabel("Falha da Máquina")
plt.ylabel("Quantidade")

# Alterar os nomes do eixo X
ax.set_xticklabels(["Sem Falha", "Com Falha"])

# Cálculo dos percentuais
total = len(df_tratado)

sem_falha = (df_tratado["falha_maquina"] == 0).sum()
com_falha = (df_tratado["falha_maquina"] == 1).sum()

perc_sem = sem_falha / total * 100
perc_com = com_falha / total * 100

# Valores sobre as barras
for barra in ax.patches:

    altura = barra.get_height()

    percentual = altura / total * 100

    ax.annotate(
        f"{int(altura)}\n({percentual:.1f}%)",
        (barra.get_x() + barra.get_width()/2, altura),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )


# Espaço para os rótulos
ax.set_ylim(0, max(df_tratado["falha_maquina"].value_counts()) * 1.15)

ax = sns.countplot(
    x="falha_maquina",
    data=df_tratado,
    palette=["green", "red"]
)

plt.show()

# Heatmap de Correlação (Pearson)

plt.figure(figsize=(9,7))

sns.heatmap(
    df_tratado.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Matriz de Correlação de Pearson")

plt.show()

# boxplot das variáveis numéricas:
# identificar os 'outliers' com o grafico boxplot

colunas = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min"
]

fig, axes = plt.subplots(3, 2, figsize=(11, 6))

axes = axes.flatten()

for i, coluna in enumerate(colunas):

    sns.boxplot(
        x=df_tratado[coluna],
        ax=axes[i]
    )

    axes[i].set_title(coluna)

# Remove o último espaço vazio
fig.delaxes(axes[5])

plt.tight_layout()

plt.show()

## analise de outliers

colunas = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min"
]

for coluna in colunas:

    Q1 = df_tratado[coluna].quantile(0.25)
    Q3 = df_tratado[coluna].quantile(0.75)

    IQR = Q3 - Q1

    inferior = Q1 - 1.5 * IQR
    superior = Q3 + 1.5 * IQR

    quantidade = (
        (df_tratado[coluna] < inferior) |
        (df_tratado[coluna] > superior)
    ).sum()

    print(f"{coluna}: {quantidade} outliers")



# Criar nova variável potência

df_tratado["potencia"] = (
    df_tratado["velocidade_rotacao_rpm"] *
    df_tratado["torque_nm"]
)

# alem da potencia criei 2 novas features [delta_temperatura] e [carga_mecanica]

# Diferença de temperatura
df_tratado["delta_temperatura"] = (
    df_tratado["temperatura_processo_k"] -
    df_tratado["temperatura_ar_k"]
)

# Carga mecânica
df_tratado["carga_mecanica"] = (
    df_tratado["torque_nm"] *
    df_tratado["desgaste_ferramenta_min"]
)


# Visualizar

df_tratado[
    ["velocidade_rotacao_rpm",
     "torque_nm",
     "potencia",
     "delta_temperatura",
     "carga_mecanica",
     ]
].head()

# confirmar a existencia de dados nulos

df_tratado.info()

## 1. Separar X e y
# variável alvo: "falha_maquina":

# Variáveis preditoras
X = df_tratado.drop(columns=["falha_maquina"])

# Variável alvo
y = df_tratado["falha_maquina"]


# Divisão treino e teste
# Utilizando 80% para treino e 20% para teste. (O stratify=y mantém a mesma proporção de falhas nos dois conjuntos)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Antes do SMOTE")
print(y_train.value_counts())

# Balanceamento
# Aplicar o SMOTE (somente no treino)

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nDepois do SMOTE")
print(y_train.value_counts())

print(X.shape)
print(y.shape)

# Conferir os tamanhos

print("Treino:", X_train.shape)
print("Teste :", X_test.shape)

# 

print("Distribuição do conjunto de treino:")

print(y_train.value_counts())

print("Após o SMOTE:")

print(y_train.value_counts())


print ("=== dados de treino ===")
print(X_train.shape)
print(y_train.shape)

print()

print ("=== dados de teste ===")
print(X_test.shape)
print(y_test.shape)





# Variáveis contínuas
colunas_continuas = [
    "temperatura_ar_k",
    "temperatura_processo_k",
    "velocidade_rotacao_rpm",
    "torque_nm",
    "desgaste_ferramenta_min",
    "potencia",
    "delta_temperatura",
    "carga_mecanica"
]

# Cópias dos conjuntos para o KNN
X_train_knn = X_train.copy()
X_test_knn = X_test.copy()

# Criar o StandardScaler
scaler = StandardScaler()

# Ajustar e transformar o treino
    # O que significa fit_transform?
    # O StandardScaler calcula: média e desvio padrão, utilizando apenas os dados de treinamento.
    # Depois transforma esses dados.

X_train_knn[colunas_continuas] = scaler.fit_transform(
    X_train_knn[colunas_continuas]
)

# Transformar o teste usando os parâmetros do treino
    # usamos  'transform' porque o conjunto de teste representa dados que o modelo nunca viu.
    # se fizermos 'fit_transform', estaríamos usando informações do teste para calcular médias e desvios, o que caracteriza Data Leakage.

    
X_test_knn[colunas_continuas] = scaler.transform(
    X_test_knn[colunas_continuas]
)

print("Escalonamento concluído!")

# K N N

# Testando vários valores de K

k_valores = [3, 5, 7]

print("========== KNN ==========")

for k in k_valores:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train_knn, y_train)

    y_pred_train = knn.predict(X_train_knn)

    y_pred_test = knn.predict(X_test_knn)

    acc_train = accuracy_score(
        y_train,
        y_pred_train
    )

    acc_test = accuracy_score(
        y_test,
        y_pred_test
    )

    print(f"\nK = {k}")

    print(f"Acurácia Treino: {acc_train:.4f}")

    print(f"Acurácia Teste : {acc_test:.4f}")

# Árvore de Decisão

profundidades = [3, 5, None]

print("========== ÁRVORE ==========")

for profundidade in profundidades:

    arvore = DecisionTreeClassifier(
        max_depth=profundidade,
        random_state=42
    )

    arvore.fit(
        X_train,
        y_train
    )

    pred_train = arvore.predict(X_train)

    pred_test = arvore.predict(X_test)

    acc_train = accuracy_score(
        y_train,
        pred_train
    )

    acc_test = accuracy_score(
        y_test,
        pred_test
    )

    print(f"\nProfundidade = {profundidade}")

    print(f"Acurácia Treino: {acc_train:.4f}")

    print(f"Acurácia Teste : {acc_test:.4f}")





# comparativo KNN

k_valores = [3, 5, 7]

resultados_knn = []

for k in k_valores:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train_knn, y_train)

    pred_train = knn.predict(X_train_knn)
    pred_test = knn.predict(X_test_knn)

    acc_train = accuracy_score(
        y_train,
        pred_train
    )

    acc_test = accuracy_score(
        y_test,
        pred_test
    )

    diferenca = abs(acc_train - acc_test)

    resultados_knn.append({
        "K": k,
        "Acurácia Treino": acc_train,
        "Acurácia Teste": acc_test,
        "Diferença": diferenca
    })

# Criar DataFrame
comparativo_knn = pd.DataFrame(resultados_knn)

# Formatar como percentual
comparativo_knn["Acurácia Treino"] = comparativo_knn["Acurácia Treino"].map(lambda x: f"{x:.2%}")
comparativo_knn["Acurácia Teste"] = comparativo_knn["Acurácia Teste"].map(lambda x: f"{x:.2%}")
comparativo_knn["Diferença"] = comparativo_knn["Diferença"].map(lambda x: f"{x:.2%}")

print("\n=== COMPARATIVO KNN ===")
print(comparativo_knn)

# melhor configuração do KNN
comparativo_numerico = pd.DataFrame(resultados_knn)

melhor = comparativo_numerico.loc[
    comparativo_numerico["Acurácia Teste"].idxmax()
]

print("\nMelhor configuração do KNN:")
print(f"K = {melhor['K']}")
print(f"Acurácia de teste = {melhor['Acurácia Teste']:.2%}")

# Comparativo Arvore de Decisão

# Valores de profundidade a serem testados
profundidades = [3, 5, None]

# Lista para armazenar os resultados
resultados_arvore = []

# Treinamento e avaliação
for profundidade in profundidades:

    arvore = DecisionTreeClassifier(
        max_depth=profundidade,
        random_state=42
    )

    arvore.fit(X_train, y_train)

    pred_train = arvore.predict(X_train)
    pred_test = arvore.predict(X_test)

    acc_train = accuracy_score(
        y_train,
        pred_train
    )

    acc_test = accuracy_score(
        y_test,
        pred_test
    )

    diferenca = abs(acc_train - acc_test)

    resultados_arvore.append({
        "Profundidade": profundidade,
        "Acurácia Treino": acc_train,
        "Acurácia Teste": acc_test,
        "Diferença": diferenca
    })

# Criar DataFrame numérico
comparativo_arvore = pd.DataFrame(resultados_arvore)

# Mostrar a tabela formatada
comparativo_formatado = comparativo_arvore.copy()

comparativo_formatado["Acurácia Treino"] = (
    comparativo_formatado["Acurácia Treino"]
    .map(lambda x: f"{x:.2%}")
)

comparativo_formatado["Acurácia Teste"] = (
    comparativo_formatado["Acurácia Teste"]
    .map(lambda x: f"{x:.2%}")
)

comparativo_formatado["Diferença"] = (
    comparativo_formatado["Diferença"]
    .map(lambda x: f"{x:.2%}")
)

print("\n=== COMPARATIVO ÁRVORE DE DECISÃO ===")
print(comparativo_formatado)

# Encontrar automaticamente a melhor configuração
melhor_arvore = comparativo_arvore.loc[
    comparativo_arvore["Acurácia Teste"].idxmax()
]

print("\n=== MELHOR CONFIGURAÇÃO ===")
print(f"Profundidade: {melhor_arvore['Profundidade']}")
print(f"Acurácia Treino: {melhor_arvore['Acurácia Treino']:.2%}")
print(f"Acurácia Teste: {melhor_arvore['Acurácia Teste']:.2%}")
print(f"Diferença: {melhor_arvore['Diferença']:.2%}")




# criar um data frame unico contendo as informacoes do resultado dos dois modelos 

# Lista para armazenar todos os resultados
comparativo_modelos = []

# ==========================
# KNN
# ==========================

k_valores = [3, 5, 7]

for k in k_valores:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train_knn, y_train)

    pred_train = knn.predict(X_train_knn)
    pred_test = knn.predict(X_test_knn)

    treino = accuracy_score(
        y_train,
        pred_train
    )

    teste = accuracy_score(
        y_test,
        pred_test
    )

    comparativo_modelos.append({
        "Modelo": "KNN",
        "Parâmetro": f"K = {k}",
        "Acurácia Treino": treino,
        "Acurácia Teste": teste,
        "Diferença": abs(treino - teste)
    })

# ==========================
# Árvore de Decisão
# ==========================

profundidades = [3, 5, None]

for profundidade in profundidades:

    arvore = DecisionTreeClassifier(
        max_depth=profundidade,
        random_state=42
    )

    arvore.fit(
        X_train,
        y_train
    )

    pred_train = arvore.predict(X_train)

    pred_test = arvore.predict(X_test)

    treino = accuracy_score(
        y_train,
        pred_train
    )

    teste = accuracy_score(
        y_test,
        pred_test
    )

    comparativo_modelos.append({
        "Modelo": "Árvore",
        "Parâmetro": f"Depth = {profundidade}",
        "Acurácia Treino": treino,
        "Acurácia Teste": teste,
        "Diferença": abs(treino - teste)
    })

# ==========================
# Criar DataFrame
# ==========================

comparativo = pd.DataFrame(comparativo_modelos)

# Classificação simples do comportamento
def classificar_modelo(linha):

    if linha["Acurácia Treino"] == 1.0:
        return "Possível Overfitting"

    elif linha["Acurácia Teste"] < 0.90:
        return "Underfitting"

    else:
        return "Bom ajuste"

comparativo["Diagnóstico"] = comparativo.apply(
    classificar_modelo,
    axis=1
)

# Guardar versão numérica
comparativo_numerico = comparativo.copy()

# Formatar percentuais para exibição
comparativo["Acurácia Treino"] = comparativo["Acurácia Treino"].map(
    lambda x: f"{x:.2%}"
)

comparativo["Acurácia Teste"] = comparativo["Acurácia Teste"].map(
    lambda x: f"{x:.2%}"
)

comparativo["Diferença"] = comparativo["Diferença"].map(
    lambda x: f"{x:.2%}"
)

print("\n========== COMPARATIVO GERAL ==========\n")

print (comparativo)
print()

print ('Descrição do Diagnósticos')
print ("Bom ajuste → diferença entre treino e teste menor que 5%.")
print ("Possível Overfitting → acurácia de treino muito alta (por exemplo, acima de 99%) e teste inferior.")
print ("Possível Underfitting → acurácia baixa tanto no treino quanto no teste.")



# ==========================
# Melhor modelo
# ==========================

melhor = comparativo_numerico.loc[
    comparativo_numerico["Acurácia Teste"].idxmax()
]

print("\n==============================")
print(" MELHOR MODELO ENCONTRADO")
print("==============================")

print(f"Modelo.............: {melhor['Modelo']}")
print(f"Parâmetro..........: {melhor['Parâmetro']}")
print(f"Acurácia Treino....: {melhor['Acurácia Treino']:.2%}")
print(f"Acurácia Teste.....: {melhor['Acurácia Teste']:.2%}")
print(f"Diferença..........: {melhor['Diferença']:.2%}")

