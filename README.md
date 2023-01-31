
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

---

**Carpeta *Regresion_linal_2*:**

- Hacemos de nuevo los procesos necesarios para preparar sobre los mismo datos del metro un nuevo modélo de predicción.

---

**Carpeta *Regresion_logistica*:**

- EDA
- Preprocesado
- Ajuste regresión logística
- Métricas regresión logística
- Decision Tree
- Random Forest

---

**Carpeta Archivos:**

- Metro_Interstate_Traffic_Volumne.csv.gz:

    - Utilizado para los notebooks de la carpeta *Regresión_lineal*  y avanzado en los archivos ---> Metro_1, metro_2, metro_3, metro_4

    - Utilizado para los notebooks de la carpeta *Regresion_lineal_2*  y avanzado en los archivos ---> Metro_A, metro_B, metro_C, metro_D_stan_enco

- travel_insurance.csv:

    - Utilizado para los notebooks de la carpeta *Regresion_logistica* y avanzado en los archivos ---> travel_1, travel_balanceado, travel_balanceado_sin_dupl, travel_enco_stand_sin_balanceo, resultados_travel_logistica.


---


**Librerías utilizadas:**

# Tratamiento de los datos
import pandas as pd
import numpy as np
import random 

# Gráficas
import matplotlib.pyplot as plt
import seaborn as sns
import sidetable 
import awoc
from tqdm import tqdm

# Estadísticos
from scipy.stats import skew
from scipy.stats import kurtosistest
from scipy.stats import kstest
from scipy import stats
import researchpy as rp
from scipy.stats import levene
import math 
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Estandarización
from sklearn.preprocessing import StandardScaler

# Codificación de las variables numéricas
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 

# Modelado y evaluación
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score

# Crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics

# Gestión datos desbalanceados
from imblearn.combine import SMOTETomek

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')

