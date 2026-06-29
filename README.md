```markdown
# 🚢 Titanic Survival Prediction - ML 
# Pipeline
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg) 
![Accuracy](https://img.shields.io/badge/Accuracy-83.24%25-green.svg)
Pipeline profesional de Machine Learning 
para predecir la supervivencia en el 
Titanic. Arquitectura modular (OOP) con 
limpieza inteligente, *feature 
engineering*, prevención de *data leakage* 
y comparación de modelos. ---
## 🔍 Análisis Exploratorio (EDA)
![Distribución de 
!Supervivencia](images/survived_distribution.png)
*El 61.6% no sobrevivió. Mujeres y 1ra 
clase tuvieron tasas drásticamente 
superiores.*
| ![Variables 
| !Categóricas](images/categorical_vs_survived.png) 
| !| 
| !![Correlación](images/correlation_matrix.png) 
| !|
|:---:|:---:|
| Variables Categóricas vs Supervivencia | 
| Matriz de Correlación |
---
## 🛠️ Metodología: Las 6 Partes Clave
### 1️⃣ Limpieza Inteligente (No es un simple 
### `fillna`)
En lugar de usar la media global para 
`Age`, se imputó con la **mediana agrupada 
por `Pclass`**: - 1ra clase → mediana ~38 
años - 3ra clase → mediana ~25 años
> 💡 **Por qué importa:** Preserva la 
> relación real entre clase social y edad, 
> evitando sesgar el modelo.
### 2️⃣ Feature Engineering Estratégico
Se crearon variables que capturan el 
contexto del viaje: ```python FamilySize = 
SibSp + Parch + 1 # Tamaño del grupo 
isAlone = 1 if FamilySize == 1 else 0 ``` 
**Hipótesis validada:** Familias pequeñas 
(2-4 personas) sobreviven más que pasajeros 
solos o familias grandes.
### 3️⃣ Pipeline con `ColumnTransformer` 
### (Anti Data Leakage)
``` Pipeline ├── StandardScaler (numéricas) 
└── OneHotEncoder (categóricas) ```
> 💡 **Por qué importa:** Los 
> transformadores se ajustan **solo con 
> datos de train**, simulando un escenario 
> real de producción.
### 4️⃣ Optimización con `GridSearchCV`
Búsqueda sistemática de hiperparámetros con 
validación cruzada estratificada de 5 folds 
+ caché inteligente con `joblib` para no 
reentrenar innecesariamente.
### 5️⃣ Métricas Más Allá del Accuracy
| Métrica | Random Forest | Gradient 
| Boosting |
|---------|:---:|:---:| Accuracy | 0.8212 | 
| **0.8324** | Recall (sobrevivientes) | 
| 0.6522 | **0.6812** | ROC-AUC | 0.8398 | 
| **0.8612** |
### 6️⃣ Interpretabilidad (Feature 
### Importance)
Se extrajeron las variables más influyentes 
para validar que el modelo tenga sentido 
histórico y de negocio, no que sea una 
"caja negra". ---
## 📈 Resultados
![Comparación de 
!Modelos](images/model_comparison.png)
🏆 **Ganador: Gradient Boosting** con 
83.24% de accuracy.
![Matriz de 
!Confusión](images/confusion_matrix.png) 
![Feature 
!Importance](images/feature_importance.png)
**Hallazgos de Negocio:** El modelo valida 
históricamente la política *"mujeres y 
niños primero"* (`Sex_male` es el predictor 
#1) y demuestra que el estatus 
socioeconómico (`Pclass_3` y `Fare`) fue 
determinante para acceder a los botes 
salvavidas. ---
## 🚀 Instalación y Uso
```bash git clone 
https://github.com/leonardoglez7/titanic-survival-prediction.git 
cd titanic-survival-prediction pip install 
-r requirements.txt
# Pipeline completo
python main.py
# Generar imágenes
python generate_images.py ```
## 📁 Estructura del Proyecto
```text ├── src/ │ ├── data_cleaning.py # 
Clase TitanicCleaner │ └── 
model_pipeline.py # Clase TitanicModel ├── 
main.py # Orquestador ├── 
generate_images.py # Visualizaciones ├── 
data/ # Datasets ├── models/ # Modelos .pkl 
└── images/ # Gráficos ```
## ⚠️ Próximos Pasos
- Extraer títulos desde `Name` (Mr, Mrs, 
Master, Dr) - Probar XGBoost / LightGBM - 
Aplicar SMOTE para maximizar *Recall* - 
Validación cruzada de 10 folds --- 
**Autor:** Leonardo González | 
[GitHub](https://github.com/leonardoglez7)
```
