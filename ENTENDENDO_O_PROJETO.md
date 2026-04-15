# 📚 Entendendo o Projeto: One Bank — Previsão de Churn Bancário

**Bem-vindo(a)!** Este guia explica o projeto inteiro de forma acessível para iniciantes.
Se você clonou este repositório e quer entender tudo — das bibliotecas às decisões de negócio — você está no lugar certo! 🎯

---

## 📖 Índice

1. [O que é este projeto?](#-o-que-é-este-projeto)
2. [Explicação das Bibliotecas (Stack)](#-explicação-das-bibliotecas-stack)
3. [Fluxo Completo do Notebook](#-fluxo-completo-do-notebook)
   - [Seção 1: Contexto e Objetivos](#seção-1-contexto-e-objetivos)
   - [Seção 2: Carregamento e Inspeção](#seção-2-carregamento-e-inspeção)
   - [Seção 3: Análise Exploratória (EDA)](#seção-3-análise-exploratória-eda)
   - [Seção 4: Feature Engineering](#seção-4-feature-engineering)
   - [Seção 5: Modelagem](#seção-5-modelagem)
   - [Seção 6: Interpretabilidade (SHAP)](#seção-6-interpretabilidade-shap)
   - [Seção 7: Impacto Financeiro (ROI)](#seção-7-impacto-financeiro-roi)
   - [Seção 8: Exportação e Simulação de Produção](#seção-8-exportação-e-simulação-de-produção)
4. [Glossário de Termos Técnicos](#-glossário-de-termos-técnicos)
5. [Principais Insights](#-principais-insights)
6. [Como Executar](#-como-executar)

---

## 🎯 O que é este projeto?

Este é um **projeto de Machine Learning** que prevê quando um cliente de banco vai **sair/desistir** (churn).

### Por que isso importa?

- **Um cliente perdido = lucro perdido**: Conquistar um cliente novo custa de **5 a 25 vezes mais** do que reter um existente.
- **Economia**: Se conseguirmos identificar quem quer sair *antes* que saia, podemos oferecer incentivos para ficar.
- **ROI real**: Este modelo **transforma uma perda de R$ 2 milhões em um lucro de R$ 655 mil**.

### Pipeline completo

O notebook segue um fluxo de ponta a ponta:

```
EDA → Feature Engineering → Comparação de Modelos → Otimização → Interpretabilidade → ROI → Produção
```

---

## 🛠️ Explicação das Bibliotecas (Stack)

Todas as bibliotecas usadas neste projeto, com explicação do que fazem e **como são usadas no código**.

---

### 📊 pandas

**O que é:** Biblioteca para trabalhar com dados em formato tabular (tipo Excel, mas muito mais poderosa).

**Por que é usada neste projeto:**
- Carregar o arquivo CSV com 10.000 clientes
- Inspecionar dados (`df.head()`, `df.info()`, `df.describe()`)
- Verificar dados faltantes (`df.isnull().sum()`)
- Criar novas colunas (Feature Engineering)
- Calcular crosstabs para análise bivariada

**Como aparece no código:**
```python
import pandas as pd

# Carregamento dos dados
df = pd.read_csv('Customer-Churn-Records.csv')

# Inspeção rápida
df.head()       # Primeiras 5 linhas
df.info()       # Tipos de dados e tamanho
df.describe()   # Estatísticas (média, min, max, etc.)

# Análise bivariada com crosstab
pd.crosstab(df['Geography'], df['Exited'], normalize='index') * 100
```

---

### 🔢 numpy

**O que é:** Biblioteca para cálculos numéricos rápidos com arrays e matrizes.

**Por que é usada neste projeto:**
- Operações matemáticas na modelagem
- Seleção de tipos de dados numéricos (`select_dtypes(include=np.number)`)
- Constante epsilon para evitar divisão por zero no Feature Engineering

**Como aparece no código:**
```python
import numpy as np

# Identificar colunas numéricas
colunas_num = X.select_dtypes(include=np.number).columns.tolist()

# Constante de segurança
epsilon = 1e-6
df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + epsilon)
```

---

### 📈 matplotlib

**O que é:** Biblioteca para criar gráficos (linhas, barras, histogramas, etc.).

**Por que é usada neste projeto:**
- Histogramas de distribuição (idade, saldo)
- Boxplots para identificar outliers
- Heatmaps de correlação e matrizes de confusão
- Gráficos de barras para comparação entre grupos
- Visualização do ROI financeiro

**Como aparece no código:**
```python
import matplotlib.pyplot as plt

# Histograma de idades
plt.hist(df['Age'], bins=30)
plt.title('Distribuição de Idades')
plt.show()

# Matriz de confusão
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão (Threshold 30%)')
plt.show()
```

---

### 🎨 seaborn

**O que é:** Funciona em cima do matplotlib, criando gráficos estatísticos mais bonitos e com menos código.

**Por que é usada neste projeto:**
- Heatmap de correlações (que detectou o problema com `Complain`)
- Boxplots por grupo (churn vs. não-churn)
- Gráficos de barras agrupados para as hipóteses
- Estilização visual dos gráficos

**Como aparece no código:**
```python
import seaborn as sns

# Heatmap de correlação que revelou Data Leakage
sns.heatmap(pd.crosstab(df['Complain'], df['Exited']), annot=True, fmt='d', cmap='Reds')

# Boxplot agrupado por churn
sns.boxplot(data=df, x='Exited', y='Age')
```

---

### 🤖 scikit-learn

**O que é:** A principal biblioteca de Machine Learning em Python. Oferece ferramentas para treinamento, avaliação e deployment de modelos.

**Por que é usada neste projeto:**
- Dividir dados em treino/teste (`train_test_split`)
- Validação cruzada estratificada (`StratifiedKFold`, `cross_val_score`)
- Calcular métricas (precision, recall, F1, AUC-ROC, accuracy)
- `OneHotEncoder` para transformar variáveis categóricas em números
- `ColumnTransformer` para aplicar transformações por tipo de coluna
- Logistic Regression e Random Forest como modelos de comparação

**Como aparece no código:**
```python
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Divisão treino/teste (80/20, estratificada)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### ⚡ XGBoost

**O que é:** Algoritmo de Machine Learning baseado em *gradient boosting* — combina múltiplas árvores de decisão fracas para criar um modelo forte.

**Por que é usada neste projeto:**
- Teve o melhor AUC-ROC: **0.8614** (vs. 0.7514 da Logistic Regression)
- Melhor F1-Score: **0.5925**
- Com threshold otimizado para 30%, atinge **71,81% de recall**
- Mais flexível para capturar padrões complexos nos dados

**Como aparece no código:**
```python
from xgboost import XGBClassifier

# Dentro do Pipeline
pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

pipeline_model.fit(X_train, y_train)
```

---

### ⚖️ imbalanced-learn (SMOTE)

**O que é:** Biblioteca para lidar com dados desbalanceados. SMOTE (*Synthetic Minority Over-sampling Technique*) cria exemplos sintéticos da classe minoritária.

**O problema:**
- O dataset tem ~20% de churn e ~80% de não-churn
- Sem tratamento, o modelo poderia aprender a sempre dizer "NÃO vai fazer churn" e acertar 80% das vezes
- Isso não ajuda! Queremos detectar os 20% que **vão** fazer churn

**Por que é usada neste projeto:**
- SMOTE gera dados sintéticos da classe minoritária (churn)
- Treina o modelo com uma distribuição equilibrada
- Integrada diretamente no Pipeline via `imblearn.pipeline.Pipeline`

**Como aparece no código:**
```python
from imblearn.pipeline import Pipeline  # Nota: Pipeline do imblearn, não do sklearn!
from imblearn.over_sampling import SMOTE

pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),    # Balanceamento dentro do pipeline
    ('classifier', XGBClassifier(...))
])
```

> ⚠️ **Detalhe importante:** Usa-se o `Pipeline` do `imblearn`, não do `sklearn`, porque o Pipeline padrão do sklearn não suporta etapas de reamostragem como o SMOTE.

---

### 🔍 SHAP

**O que é:** SHAP (*SHapley Additive exPlanations*) explica **quais variáveis** foram importantes para cada previsão do modelo, tornando a "caixa preta" transparente.

**Por que é usada neste projeto:**
- XGBoost é uma "caixa preta" — não sabemos por que previu cada coisa
- SHAP mostra o ranking global de importância das variáveis
- Permite explicar decisões individuais ("Este cliente vai sair porque tem 55 anos e satisfação 1")
- Essencial para confiança do negócio no modelo

**Como aparece no código:**
```python
import shap

# Acessar o modelo dentro do pipeline
model_step = pipeline_model.named_steps['classifier']
preprocessor_step = pipeline_model.named_steps['preprocessor']

# Transformar dados para formato numérico
X_test_transformed = preprocessor_step.transform(X_test)

# SHAP
explainer = shap.TreeExplainer(model_step)
shap_values = explainer.shap_values(X_test_transformed)

# Ranking global de importância
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_cols, plot_type="bar")
```

---

### 💾 joblib

**O que é:** Salva objetos Python (como modelos treinados) em arquivos `.pkl` para reutilização posterior.

**Por que é usada neste projeto:**
- Treinar um modelo leva tempo e recursos
- Depois de treinar, salvamos o pipeline inteiro para uso em produção
- Qualquer sistema pode carregar o `.pkl` e fazer previsões sem retreinar

**Como aparece no código:**
```python
import joblib

# Salvar o pipeline completo (preprocessamento + SMOTE + modelo)
joblib.dump(pipeline_model, 'pipeline_churn_v1.pkl')

# Carregar em produção
pipeline_carregado = joblib.load('pipeline_churn_v1.pkl')
probabilidade = pipeline_carregado.predict_proba(dados_novos)[0][1]
```

---

## 🔄 Fluxo Completo do Notebook

O notebook `One_Bank_Churn_Prediction.ipynb` tem **94 células** organizadas em 8 seções. Aqui explicamos cada uma.

---

### Seção 1: Contexto e Objetivos

**O que é feito:** Define o problema de negócio, as perguntas a responder e as métricas de sucesso.

**Perguntas de Negócio:**

| # | Pergunta |
|---|----------|
| 1 | Quais características dos clientes estão mais associadas à alta probabilidade de churn? |
| 2 | Clientes com Score de Crédito mais baixo têm propensão maior a sair? |
| 3 | Existe diferença nas taxas de churn entre França, Alemanha e Espanha? |
| 4 | Clientes com saldo zerado têm maior probabilidade de sair? |
| 5 | Membros ativos têm realmente menor risco de churn? |

**Métricas de Sucesso:**

- **Métrica principal:** Recall — minimizar Falsos Negativos (clientes que vão sair mas o modelo não detecta)
- **Métrica secundária:** AUC-ROC — capacidade geral de distinguir entre classes

> 💡 **Por que Recall?** Porque o custo de perder um cliente (Falso Negativo) é muito maior do que o custo de abordar um cliente que ficaria de qualquer forma (Falso Positivo).

---

### Seção 2: Carregamento e Inspeção

**O que é feito:**
- Importação de todas as bibliotecas
- Carregamento do dataset CSV
- Inspeção inicial com `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()`

**Dados do Dataset:**

| Informação | Valor |
|------------|-------|
| Linhas (clientes) | 10.000 |
| Colunas (variáveis) | 18 |
| Valores faltantes | 0 (dados limpos!) |
| Fonte | [Kaggle — Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn) |

**Principais Variáveis:**

| Variável | Tipo | Significado |
|----------|------|-------------|
| `CreditScore` | Numérica | Score de crédito (350–850) |
| `Geography` | Categórica | País (França, Alemanha, Espanha) |
| `Gender` | Categórica | Gênero (Male/Female) |
| `Age` | Numérica | Idade em anos |
| `Tenure` | Numérica | Anos como cliente do banco |
| `Balance` | Numérica | Saldo na conta |
| `NumOfProducts` | Numérica | Quantidade de produtos do banco |
| `HasCrCard` | Binária | Possui cartão de crédito (0/1) |
| `IsActiveMember` | Binária | Membro ativo (0/1) |
| `EstimatedSalary` | Numérica | Salário estimado |
| `Satisfaction Score` | Numérica | Satisfação (1–5) |
| `Point Earned` | Numérica | Pontos acumulados |
| `Card Type` | Categórica | Tipo de cartão |
| `Complain` | Binária | Reclamou (0/1) — ⚠️ removida por Data Leakage |
| `Exited` | Binária | **ALVO**: 0 = Ficou, 1 = Saiu (churn) |

---

### Seção 3: Análise Exploratória (EDA)

**O que é feito:** Entender os dados profundamente antes de treinar qualquer modelo. Dividida em 6 subseções.

#### 3.1 — Análise Univariada

Cada variável é analisada isoladamente com histogramas, boxplots e gráficos de barras.

**Destaques:**
- **Distribuição de Churn:** ~20% churn vs. ~80% não-churn (dados desbalanceados)
- **Idade:** Média de 38,9 anos. Mínimo 18, máximo 92
- **Saldo:** Cerca de 50% dos clientes têm saldo **zero** ("clientes fantasmas")
- **Geografia:** França (50%), Alemanha (25%), Espanha (25%)
- **Gênero:** Masculino (54,6%), Feminino (45,4%)

#### 3.2 — Heatmap de Correlação

Um heatmap com a correlação de todas as variáveis numéricas com `Exited` (churn).

> 🚨 Foi aqui que o **Data Leakage** com a variável `Complain` foi detectado!

#### 3.3 — 🚨 Detecção de Data Leakage

**O que aconteceu:**
A variável `Complain` apresentou correlação de **~99,57%** com `Exited`.

**A crosstab revelou uma classificação quase perfeita:**
- Verdadeiros negativos: 7.952
- Verdadeiros positivos: 2.034
- Falsos positivos: 4
- Falsos negativos: 10

**Por que isso é um problema?**
- Acurácia de ~99% é irrealista — parece "mágico" demais
- `Complain` não é uma **causa** do churn, é uma **consequência** (ou indicador simultâneo)
- Em produção, essa variável não existiria no momento da previsão
- Se usada, o modelo teria performance inflada no treino mas **falharia em dados novos**

**Solução aplicada:**
```python
colunas_para_remover = ['RowNumber', 'CustomerId', 'Surname', 'Complain']
df_feat = df_feat.drop(colunas_para_remover, axis=1)
```

> 💡 **Lição:** Sempre investigue variáveis com correlação muito alta com o alvo. Data Leakage é um dos erros mais comuns e perigosos em Machine Learning.

#### 3.4 — Análise Bivariada (Hipóteses Testadas)

6 hipóteses de negócio foram testadas com crosstabs e gráficos:

| Hipótese | Resultado |
|----------|-----------|
| Pontos acumulados influenciam no churn? | ✅ Analisado |
| Idade influencia no churn? | ✅ SIM — Clientes 45+ têm risco muito maior |
| Membros ativos têm menor risco? | ✅ SIM — Inativos saem mais |
| Alemanha tem mais churn? | ✅ SIM — ~32% vs. ~16% em França e Espanha |
| Gênero influencia? | ✅ Mulheres têm taxa de churn ligeiramente maior |
| Saldo afeta fidelidade? | ✅ SIM — Clientes com saldo zero tendem a sair |

#### 3.5 — Identificação de Outliers

Boxplots detalhados para todas as variáveis numéricas:

- **CreditScore:** Outliers abaixo do bigode inferior (scores muito baixos)
- **Age:** Outliers acima do bigode superior (idades muito altas)
- **NumOfProducts:** Clientes com 3–4 produtos são outliers
- **Balance:** Distribuição bimodal (muitos zeros e um pico no meio)

#### 3.6 — Principais Achados

Resumo consolidado de todas as descobertas da EDA antes de partir para a modelagem.

---

### Seção 4: Feature Engineering

**O que é:** Criar novas variáveis a partir das existentes para que o modelo capture padrões mais complexos.

> **Nota Técnica:** Uma constante `epsilon` (1e-6) é usada em todas as divisões para evitar `ZeroDivisionError`.

**As 5 Features Criadas:**

| # | Feature | Fórmula | Lógica de Negócio |
|---|---------|---------|-------------------|
| 1 | `CreditScoreGivenAge` | `CreditScore / Age` | Um score de 600 é bom para um jovem de 25 anos, mas ruim para uma pessoa de 70. Normaliza o score pela idade. |
| 2 | `HasBalance` | `1 se Balance > 0, senão 0` | Flag binária. Clientes com saldo zero são "fantasmas" que tendem a sair. |
| 3 | `PointsPerProduct` | `Point Earned / NumOfProducts` | Engajamento real. 1000 pontos com 1 produto = super engajado! |
| 4 | `BalanceSalaryRatio` | `Balance / EstimatedSalary` | Capacidade de poupança. Quanto do salário o cliente consegue guardar? |
| 5 | `TenureByAge` | `Tenure / Age` | Fidelidade proporcional. 10 anos de banco aos 30 é bem diferente de 10 anos aos 70. |

**Código no notebook:**
```python
epsilon = 1e-6

df_feat['CreditScoreGivenAge'] = df_feat['CreditScore'] / (df_feat['Age'] + epsilon)
df_feat['HasBalance'] = df_feat['Balance'].apply(lambda x: 1 if x > 0 else 0)
df_feat['PointsPerProduct'] = df_feat['Point Earned'] / (df_feat['NumOfProducts'] + epsilon)
df_feat['BalanceSalaryRatio'] = df_feat['Balance'] / (df_feat['EstimatedSalary'] + epsilon)
df_feat['TenureByAge'] = df_feat['Tenure'] / (df_feat['Age'] + epsilon)
```

---

### Seção 5: Modelagem

Dividida em 4 etapas: preparação, comparação, pipeline final e otimização.

#### 5.1 — Preparação dos Dados

**O que é feito:**
- Remoção de IDs (`RowNumber`, `CustomerId`, `Surname`) e `Complain` (Data Leakage)
- Separação de variáveis preditoras (`X`) e alvo (`y`)
- Identificação de colunas categóricas (`Geography`, `Gender`, `Card Type`) e numéricas
- Divisão treino/teste: **80/20**, estratificada

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

> 💡 **`stratify=y`** garante que a proporção 80/20 de churn vs. não-churn seja mantida tanto no treino quanto no teste.

#### 5.2 — Comparação de Modelos (Cross-Validation)

3 modelos avaliados com **validação cruzada estratificada (5-fold)**, todos usando o mesmo pipeline (preprocessamento + SMOTE):

| Modelo | Recall | F1-Score | AUC-ROC |
|--------|--------|----------|---------|
| Logistic Regression | **0.7252** ± 0.0161 | 0.4809 ± 0.0115 | 0.7514 ± 0.0083 |
| Random Forest | 0.4945 ± 0.0081 | 0.5903 ± 0.0082 | 0.8509 ± 0.0089 |
| **XGBoost** | 0.5025 ± 0.0098 | **0.5925** ± 0.0120 | **0.8614** ± 0.0079 |

#### 5.3 — Pipeline Final (XGBoost)

**Por que XGBoost venceu?**

Logistic Regression teve o maior Recall bruto (72,5%), porém com F1 e AUC-ROC significativamente inferiores. O XGBoost:

1. Tem o melhor **AUC-ROC** (0.86) — melhor capacidade de distinguir classes
2. Com **threshold ajustado** de 50% → 30%, atinge Recall equivalente (71,8%)
3. Mantém **melhor F1-Score** — menos falsos positivos
4. Menos alertas errados = menor custo operacional em produção

**Estrutura do Pipeline:**

```
┌──────────────────┐     ┌───────────┐     ┌─────────────────┐
│  Preprocessor    │     │   SMOTE   │     │   XGBClassifier │
│  (OneHotEncoder) │ ──▶ │ (Balance) │ ──▶ │   (Predição)    │
└──────────────────┘     └───────────┘     └─────────────────┘
```

```python
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), colunas_cat)
], remainder='passthrough')

pipeline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])
```

#### 5.4 — Otimização de Threshold

**O que é Threshold?** É o ponto de corte para decidir "churn" vs. "não-churn".

**Padrão (50%):**
```
Se Probabilidade > 50% → Classifica como CHURN
Se Probabilidade ≤ 50% → Classifica como NÃO-CHURN
```

**O problema:** Com 50%, muitos churners reais não são detectados (Recall baixo).

**Solução — Threshold Otimizado (30%):**
```
Se Probabilidade > 30% → Classifica como CHURN
Se Probabilidade ≤ 30% → Classifica como NÃO-CHURN
```

**Resultado com threshold de 30%:**

| Métrica | Valor |
|---------|-------|
| Acurácia | 83,15% |
| **Recall** | **71,81%** ✅ |
| Precisão | 56,89% |

> 💡 **Por que funciona?** Baixar o threshold faz o modelo "errar para o lado de alertar" — mais clientes em risco são detectados. Sacrificamos um pouco de precisão, mas é uma troca que vale a pena porque o custo de perder um cliente é muito maior que o custo de abordar um que ficaria.

---

### Seção 6: Interpretabilidade (SHAP)

**O que é:** SHAP explica a contribuição de **cada variável** para as previsões do modelo.

**Como funciona no código:**
1. Extrai o modelo XGBoost de dentro do pipeline
2. Transforma os dados de teste para formato numérico
3. Calcula os SHAP values para cada previsão
4. Gera um gráfico de ranking global de importância

![SHAP Feature Importance](shap_importance.png)

**Ranking de Importância Global (SHAP):**
As variáveis mais importantes para as decisões do modelo, do mais para o menos importante.

**Exemplo de interpretação individual:**
```
Cliente: João, 58 anos, saldo R$ 0, satisfação = 2

Influências que AUMENTAM o risco de churn:
  ➕ Age = 58 (acima de 45 anos)
  ➕ Balance = 0 (cliente "fantasma")
  ➕ Satisfaction Score = 2 (insatisfeito)

Influências que DIMINUEM o risco:
  ➖ IsActiveMember = 1 (usa o banco)
  ➖ Tenure = 8 (cliente de longa data)

PREVISÃO FINAL: Alta probabilidade de CHURN
```

---

### Seção 7: Impacto Financeiro (ROI)

**O que é:** Traduzir a performance do modelo em valor financeiro para o banco.

**Premissas conservadoras:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| LTV médio | R$ 5.000 | Lucro vitalício estimado por cliente |
| Custo de retenção | R$ 150 | Campanha (desconto, contato, tempo de gerente) |
| Taxa de sucesso | 50% | Estimativa conservadora de clientes salvos |

**Cálculo (sobre o conjunto de teste):**

```
Cenário A — SEM modelo (reativo):
  Clientes que saem (TP + FN) × LTV = Prejuízo total

Cenário B — COM modelo (proativo):
  Custo da campanha = Alertados (TP + FP) × R$ 150
  Clientes salvos   = TP × 50% (taxa de sucesso)
  Receita salva      = Clientes salvos × R$ 5.000
  Lucro líquido      = Receita salva − Custo da campanha
```

**Resultado:**

| Cenário | Resultado |
|---------|-----------|
| 🔴 Sem modelo (reativo) | -R$ 2.040.000 em clientes perdidos |
| 🟢 Com modelo (proativo) | **+R$ 655.250** de resultado líquido |

> A diferença entre os dois cenários é o **valor que o modelo gera para o negócio**.

![Impacto Financeiro](roi_chart.png)

---

### Seção 8: Exportação e Simulação de Produção

**O que é feito:**
1. O pipeline completo (preprocessamento + SMOTE + modelo) é salvo com `joblib`
2. Uma função `preparar_dados_backend()` é criada para uso em produção
3. Um teste final simula uma previsão real

#### Salvando o Modelo

```python
joblib.dump(pipeline_model, 'pipeline_churn_v1.pkl')
```

O arquivo `pipeline_churn_v1.pkl` contém **tudo**: OneHotEncoder, SMOTE e XGBoost. Basta carregar e usar.

#### Função de Produção

```python
def preparar_dados_backend(json_input):
    """
    Recebe JSON bruto (como viria do sistema do banco),
    aplica a engenharia de features e entrega ao Pipeline.
    """
    df = pd.DataFrame([json_input])
    epsilon = 1e-6

    # Feature Engineering
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + epsilon)
    df['TenureByAge'] = df['Tenure'] / (df['Age'] + epsilon)
    df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + epsilon)
    df['PointsPerProduct'] = df['Point Earned'] / (df['NumOfProducts'] + epsilon)
    df['HasBalance'] = df['Balance'].apply(lambda x: 1 if x > 0 else 0)

    # Padronização de texto
    df['Geography'] = df['Geography'].astype(str).str.capitalize()
    df['Gender'] = df['Gender'].astype(str).str.capitalize()
    df['Card Type'] = df['CardType'].astype(str).str.upper()

    # Seleciona colunas esperadas pelo Pipeline
    return df[cols_entrada]
```

#### Teste Final

```python
cliente_perigoso = {
    "CreditScore": 350, "Geography": "Germany", "Gender": "Female",
    "Age": 55, "Tenure": 1, "Balance": 150000.00, "NumOfProducts": 1,
    "HasCrCard": 1, "IsActiveMember": 0, "EstimatedSalary": 50000.00,
    "Satisfaction Score": 1, "Point Earned": 200, "CardType": "GOLD"
}

# 1. Prepara os dados
df_input = preparar_dados_backend(cliente_perigoso)

# 2. Faz a previsão
probabilidade = pipeline_model.predict_proba(df_input)[0][1]

# 3. Aplica threshold otimizado
if probabilidade >= 0.30:
    print(f"⚠️ ALERTA: RISCO DE CHURN — {probabilidade:.2%}")
else:
    print(f"✅ Cliente Seguro — {probabilidade:.2%}")

# Resultado: Probabilidade 91.85% → ALERTA: RISCO DE CHURN
```

---

## 📚 Glossário de Termos Técnicos

| Termo | Significado |
|-------|-------------|
| **Churn** | Cliente que cancela ou sai do banco. A taxa de churn é a porcentagem de clientes perdidos em um período. |
| **Recall** (Sensibilidade) | Dos clientes que **realmente** fizeram churn, quantos o modelo **detectou**? Fórmula: `TP / (TP + FN)`. |
| **Precision** (Precisão) | Dos clientes que o modelo **disse** que fariam churn, quantos **realmente** fizeram? Fórmula: `TP / (TP + FP)`. |
| **F1-Score** | Média harmônica entre Recall e Precision. Útil quando as duas métricas são importantes. |
| **AUC-ROC** | Área sob a curva ROC. Mede a capacidade geral do modelo de separar as duas classes (churn vs. não-churn). Quanto mais perto de 1, melhor. |
| **Data Leakage** | Quando informação do futuro "vaza" para os dados de treino, inflando artificialmente a performance. O modelo parece ótimo no treino, mas falha em produção. |
| **SMOTE** | *Synthetic Minority Over-sampling Technique*. Cria exemplos sintéticos da classe minoritária para equilibrar o dataset. |
| **Threshold** (Limiar) | Ponto de corte para transformar uma probabilidade em decisão binária. Padrão = 50%, otimizado neste projeto = 30%. |
| **SHAP** | *SHapley Additive exPlanations*. Método que explica quanto cada variável contribuiu para uma previsão específica. |
| **Cross-Validation** | Técnica que divide os dados em múltiplas partes (folds) para treinar e testar o modelo várias vezes, dando uma estimativa mais robusta de performance. |
| **Pipeline** | Sequência encadeada de etapas (preprocessamento → balanceamento → modelo) que garante consistência entre treino e produção. |
| **Feature** | Uma variável ou coluna do dataset usada como entrada para o modelo. |
| **Feature Engineering** | Processo de criar novas features a partir das existentes para melhorar a performance do modelo. |
| **Overfitting** | Quando o modelo "memoriza" os dados de treino mas falha em dados novos. Performance alta no treino, baixa no teste. |
| **Underfitting** | Quando o modelo é simples demais e não consegue aprender nem os padrões do treino. Performance ruim em tudo. |
| **LTV** | *Lifetime Value*. Lucro total que um cliente gera para a empresa ao longo do relacionamento. |
| **ROI** | *Return on Investment*. Retorno sobre o investimento. Mede se o projeto gerou mais valor do que custou. |
| **TP / FP / TN / FN** | True Positive, False Positive, True Negative, False Negative — os 4 resultados possíveis de uma classificação binária (ver matriz de confusão). |
| **Gradient Boosting** | Técnica de ensemble que combina múltiplas árvores de decisão fracas sequencialmente, onde cada árvore corrige os erros da anterior. |
| **OneHotEncoder** | Transforma variáveis categóricas (texto) em colunas binárias. Ex.: `Geography=Germany` → `Geography_Germany=1, Geography_France=0, Geography_Spain=0`. |
| **Estratificação** | Garantir que a proporção das classes seja mantida ao dividir os dados (treino/teste) ou ao fazer cross-validation. |

---

## 🎯 Principais Insights

### 📊 Padrões de Churn Identificados

| Padrão | Evidência |
|--------|-----------|
| 👴 **Idade** | Clientes acima de 45 anos têm risco significativamente maior de churn |
| 😠 **Satisfação** | Scores de satisfação baixos (1–2) correlacionam com mais churn |
| 🌍 **Geografia** | Alemanha (~32%) tem taxa de churn ~2x maior que França e Espanha (~16%) |
| 💤 **Atividade** | Membros inativos (`IsActiveMember=0`) saem muito mais |
| 💰 **Saldo** | ~50% dos clientes têm saldo zero — esses "clientes fantasmas" tendem a sair |
| 📊 **Score de Crédito** | Scores baixos (350–500) correlacionam com mais churn |
| 👩 **Gênero** | Mulheres têm taxa de churn ligeiramente superior |

### 💡 Recomendações de Negócio

1. **Foque em clientes 45+**: Criar programas de fidelização específicos para essa faixa etária
2. **Melhore a satisfação**: Atenção especial ao customer service para clientes com score 1–2
3. **Reative os "fantasmas"**: Clientes com saldo zero precisam de incentivos para engajar
4. **Investigue o mercado alemão**: Por que a Alemanha tem churn 2x maior? Pode ser cultural, concorrência, ou produto
5. **Use o modelo proativamente**: Antes que o cliente saia, ofereça um incentivo de retenção (~R$ 150)

---

## 🚀 Como Executar

### Google Colab (recomendado)

Abra o notebook no Colab e execute todas as células. As dependências já vêm pré-instaladas.

### Localmente

```bash
pip install -r requirements.txt
```

1. Baixe o dataset do [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
2. Abra o notebook no Jupyter
3. Execute todas as células sequencialmente

### Fazer novas previsões

```python
import joblib

# Carrega o pipeline salvo
pipeline = joblib.load('pipeline_churn_v1.pkl')

# Prepara os dados do cliente (aplicar feature engineering)
df_input = preparar_dados_backend(dados_do_cliente)

# Faz a previsão
probabilidade = pipeline.predict_proba(df_input)[0][1]

# Aplica o threshold otimizado
if probabilidade >= 0.30:
    print(f"⚠️ RISCO DE CHURN: {probabilidade:.2%}")
else:
    print(f"✅ Cliente seguro: {probabilidade:.2%}")
```

---

## 📚 Referências

- [Scikit-learn — Documentação](https://scikit-learn.org/)
- [XGBoost — Documentação](https://xgboost.readthedocs.io/)
- [SHAP — GitHub](https://github.com/shap/shap)
- [Imbalanced-learn — Documentação](https://imbalanced-learn.org/)
- [Dataset Original — Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

---

## 👨‍💻 Autores

- **Antonio Sergio Castro de Carvalho Junior** ([@ASCCJR](https://github.com/ASCCJR))
- **Pedro Camargo** ([@Pdrnho](https://github.com/Pdrnho))

---

**Dúvidas?** Abra uma [issue](https://github.com/ASCCJR/One-Bank-Churn-Prediction/issues) no repositório! 🎉
