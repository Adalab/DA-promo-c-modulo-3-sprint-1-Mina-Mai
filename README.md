
### Repositorio con ejercicios de pair programing del módulo 3-sprint 1 del Bootcamp de Data Analyst PromoC

**Marina Rodríguez y Marina Ruiz**

---

**Carpeta *Regresion_linal*:**

- Intro a maching learning
- Test estadísticos
- Covarianza y correlación
- Asunciones regresión lineal 
- Normalización
- Estandarización
- Anova
- Encoding
- Regresión Lineal Métricas
- Decision Tree
- Random Forest

En estos archivos encontramos un acercamiento a un modelo predictivo de regresión lineal, donde intentaremos predecir el número de personas que usan el metro de Wisconsin y si este está influido por el clima.

Nuestra H0 será que el clima no afecta al volumen de personas que cogen el metro.
Y nuestra H1 será que el clima si afecta al volumen de personas que cogen el metro.
A través de estos jupyters veremos el proceso que se seguiría en una regresión lineal normal y su alternativa de usar modelos como el Decision Tree y el Random Forest. 
---

**Carpeta *Regresion_linal_2*:**

En los archivos de esta carpeta volveremos sobre los primeros pasos del proyecto anterior de regresion_lineal y replantearemos nuestra hipótesis alternativa con la finalidad de obtener mejores resultados en nuestras predicciones sobre el número de personas que cogen el metro de Wisconsin.

Nuestra H0 será que la hora y el día no influyen en la gente que acude al metro.
Y nuestra H1 será que la hora y el día si influye en la gente que acude al metro.

---

**Carpeta *Regresion_logistica*:**

- EDA
- Preprocesado
- Ajuste regresión logística
- Métricas regresión logística
- Decision Tree
- Random Forest

En estos archivos encontraremos el proceso a seguir en un modelo predictivo de regresión logística, donde intentaremos predecir si los clientes de una aseguradora de agencias de viajes van a reclamar o no el seguro comprado.
Seguiremos los pasos que llevan a una regresión logística y haremos la comparativa con los modelos de Decision Tree y Random Forest.

---

**Carpeta *Archivos*:**

- Metro_Interstate_Traffic_Volumne.csv.gz:

    - Utilizado para los notebooks de la carpeta *Regresión_lineal*  y avanzado en los archivos ---> Metro_1, metro_2, metro_3, metro_4

    - Utilizado para los notebooks de la carpeta *Regresion_lineal_2*  y avanzado en los archivos ---> Metro_A, metro_B, metro_C, metro_D_stan_enco

- travel_insurance.csv:

    - Utilizado para los notebooks de la carpeta *Regresion_logistica* y avanzado en los archivos ---> travel_1, travel_balanceado, travel_balanceado_sin_dupl, travel_enco_stand_sin_balanceo, resultados_travel_logistica.


---


**Librerías utilizadas:**

- Tratamiento de los datos
    - pandas as pd
    - numpy as np
    - random 
    - sidetable 
    - awoc
    - tqdm

- Gráficas
    - matplotlib.pyplot as plt
    - seaborn as sns

- Estadísticos
    - scipy.stats import skew
    - from scipy.stats import kurtosistest
    - from scipy.stats import kstest
    - from scipy import stats
    - import researchpy as rp
    - from scipy.stats import levene
    - import math 
    - from sklearn.preprocessing import MinMaxScaler
    - import statsmodels.api as sm
    - from statsmodels.formula.api import ols

- Estandarización
    - from sklearn.preprocessing import StandardScaler

- Codificación de las variables numéricas
    - from sklearn.preprocessing import LabelEncoder 
    - from sklearn.preprocessing import OneHotEncoder 

- Modelado y evaluación
    - from sklearn.model_selection import train_test_split
    - from sklearn.linear_model import LinearRegression
    - from sklearn.tree import DecisionTreeClassifier
    - from sklearn.tree import DecisionTreeRegressor
    - from sklearn.ensemble import RandomForestClassifier
    - from sklearn import tree
    - from sklearn.model_selection import GridSearchCV
    - from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score

- Crossvalidation
    - from sklearn.model_selection import cross_val_score
    - from sklearn.model_selection import cross_validate
    - from sklearn import metrics

- Gestión datos desbalanceados
    - from imblearn.combine import SMOTETomek

- Configuración warnings
    - import warnings

