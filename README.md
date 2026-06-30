# 🚢 Titanic Survival Prediction - ML Pipeline

[

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

](https://www.python.org/)
[

![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange.svg)

](https://scikit-learn.org/)
[

![License](https://img.shields.io/badge/License-MIT-green.svg)

](LICENSE)

Pipeline profesional de Machine Learning para predecir la supervivencia de pasajeros del Titanic, construido con una arquitectura modular orientada a objetos que compara Random Forest y Gradient Boosting, alcanzando un **83.24% de accuracy** y un **ROC-AUC de 0.8612**.

---

## 📑 Tabla de Contenidos

- [Descripción del Proyecto](#-descripción-del-proyecto)
- [Análisis Exploratorio (EDA)](#-análisis-exploratorio-eda)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Interpretación de Negocio](#-interpretación-de-negocio)
- [Instalación y Uso](#-instalación-y-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Dependencias](#-dependencias)
- [Próximos Pasos](#-próximos-pasos)
- [Autor y Licencia](#-autor-y-licencia)

---

## 🎯 Descripción del Proyecto

Este proyecto implementa un pipeline completo de Machine Learning para predecir si un pasajero del Titanic sobrevivió o no, basándose en variables como clase social, sexo, edad, tarifa pagada y puerto de embarque.

**¿Por qué es un proyecto profesional?**

- **Programación Orientada a Objetos (OOP):** dos clases principales (`TitanicCleaner` y `TitanicModel`) que encapsulan toda la lógica de limpieza y modelado.
- **Pipeline de scikit-learn:** todo el preprocesamiento (escalado, codificación) está integrado dentro de un `Pipeline`, evitando data leakage entre entrenamiento y prueba.
- **Comparación de modelos:** evaluación rigurosa de Random Forest vs Gradient Boosting con métricas estándar de clasificación.
- **Optimización de hiperparámetros:** uso de `GridSearchCV` para maximizar el rendimiento del modelo final.

**Stack tecnológico:**

- Python 3.8+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

---

## 📊 Análisis Exploratorio (EDA)

### Distribución de supervivencia



![Distribución de supervivencia](images/survived_distribution.png)



- El **61.6%** de los pasajeros no sobrevivió, frente al 38.4% que sí.
- Confirma un dataset moderadamente desbalanceado, relevante a la hora de elegir métricas de evaluación.

### Variables numéricas vs Supervivencia



![Numéricas vs Supervivencia](images/numeric_vs_survived.png)



- Los pasajeros que pagaron tarifas (`Fare`) más altas tuvieron mayor tasa de supervivencia.
- La edad (`Age`) muestra una ligera tendencia: niños y jóvenes presentan mejor tasa de supervivencia.

### Variables categóricas vs Supervivencia



![Categóricas vs Supervivencia](images/categorical_vs_survived.png)



- `Pclass`: los pasajeros de primera clase sobrevivieron en mayor proporción que los de tercera.
- `Sex`: las mujeres tuvieron una tasa de supervivencia notablemente mayor que los hombres.
- `Embarked`: el puerto de embarque muestra diferencias menores pero consistentes.

### Matriz de correlación



![Matriz de correlación](images/correlation_matrix.png)



- `Fare` y `Pclass` muestran una correlación negativa fuerte (a mayor clase numérica, menor tarifa).
- `Sex` (codificado) es una de las variables con mayor correlación respecto a `Survived`.

---

## 🧪 Metodología

### 1. Limpieza de datos

Implementada en la clase `TitanicCleaner`:

- **Imputación de `Age`:** se imputa por la mediana agrupada según `Pclass`, capturando mejor la distribución real de edades por clase social.
- **Imputación de `Embarked`:** se rellena con la moda (valor más frecuente).
- Eliminación de columnas no informativas o con exceso de valores nulos (`Cabin`, `Ticket`, `Name` en su forma original).

### 2. Feature Engineering

- **`FamilySize`**: combinación de `SibSp` + `Parch` + 1, para capturar el tamaño total del grupo familiar.
- **`isAlone`**: variable binaria que indica si el pasajero viajaba solo.

### 3. Preprocesamiento

- **`StandardScaler`** para variables numéricas.
- **`OneHotEncoder`** para variables categóricas.
- Todo integrado en un único `Pipeline` de scikit-learn, garantizando que el preprocesamiento se ajuste solo con datos de entrenamiento y se aplique de forma consistente al test, **previniendo data leakage**.

### 4. Modelos comparados

- **Random Forest Classifier**
- **Gradient Boosting Classifier**

### 5. Optimización

- Búsqueda de hiperparámetros mediante `GridSearchCV` con validación cruzada, optimizando sobre métricas de clasificación relevantes (accuracy / ROC-AUC).

---

## 📈 Resultados

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest | 0.8212 | 0.8491 | 0.6522 | 0.7377 | 0.8398 |
| **Gradient Boosting** | **0.8324** | **0.8605** | **0.6812** | **0.7603** | **0.8612** |

> 🏆 **Gradient Boosting** fue seleccionado como modelo final por obtener el mejor desempeño en todas las métricas evaluadas.



![Comparación de modelos](images/model_comparison.png)





![Matriz de confusión](images/confusion_matrix.png)





![Importancia de variables](images/feature_importance.png)



**¿Qué significa cada métrica?**

- **Accuracy:** proporción total de predicciones correctas.
- **Precision:** de los pasajeros predichos como sobrevivientes, cuántos realmente lo fueron.
- **Recall:** de los pasajeros que realmente sobrevivieron, cuántos fueron correctamente identificados.
- **F1-Score:** media armónica entre precision y recall, útil ante clases desbalanceadas.
- **ROC-AUC:** capacidad del modelo para distinguir entre clases en distintos umbrales de decisión.

---

## 💡 Interpretación de Negocio

- **`Sex_male` es el predictor #1:** confirma el principio histórico de **"mujeres y niños primero"** aplicado durante la evacuación del Titanic. Ser hombre redujo drásticamente la probabilidad de supervivencia.
- **`Pclass_3` es altamente relevante:** los pasajeros de tercera clase tenían menor acceso a botes salvavidas y cubiertas superiores, lo que se traduce en menor tasa de supervivencia.
- **Validación histórica del modelo:** los patrones detectados por el modelo coinciden con los relatos históricos del desastre, lo que da credibilidad y coherencia narrativa a los resultados obtenidos.

---

## ⚙️ Instalación y Uso

```bash
git clone https://github.com/leonardoglez7/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
python main.py
python generate_images.py