
### Repositorio con ejercicios de pair programing del módulo 3-sprint 1 promo C 

**Marina Rodríguez y Marina Ruiz**

En este repositorio encontraremos los ejercicios realizados en el pair programing del bootcamp de Data Analyst Promo C de Adalab, siguiendo las metodologías de Scrum de la filosofía Agile.

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
    - pandas
    - numpy
    - random 
    - sidetable 
    - awoc
    - tqdm

- Gráficas
    - matplotlib.pyplot
    - seaborn
    - warnings

- Estadísticos
    - scipy.stats método skew
    - scipy.stats método kurtosistest
    - scipy.stats método kstest
    - scipy método stats
    - researchpy 
    - scipy.stats método levene
    - math 
    - sklearn.preprocessing método MinMaxScaler
    - statsmodels.api
    - statsmodels.formula.api método ols

- Estandarización
    - sklearn.preprocessing método StandardScaler

- Codificación de las variables numéricas
    - sklearn.preprocessing método LabelEncoder 
    - sklearn.preprocessing método OneHotEncoder 

- Modelado y evaluación
    - sklearn.model_selection método train_test_split
    - sklearn.linear_model método LinearRegression
    - sklearn.tree método DecisionTreeClassifier
    - sklearn.tree método DecisionTreeRegressor
    - sklearn.ensemble método RandomForestClassifier
    - sklearn método tree
    - sklearn.model_selection método GridSearchCV
    - sklearn.metrics método r2_score, mean_squared_error, mean_absolute_error
    - sklearn.linear_model método LogisticRegression
    - sklearn.metrics método confusion_matrix, accuracy_score, precision_score, recall_score, f1_score , cohen_kappa_score, roc_curve,roc_auc_score

- Crossvalidation
    - sklearn.model_selection método cross_val_score
    - sklearn.model_selection método cross_validate
    - sklearn método metrics

- Gestión datos desbalanceados
    - imblearn.combine método SMOTETomek


