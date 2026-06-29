import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_cleaning import TitanicCleaner
from src.model_pipeline import TitanicModel
from sklearn.metrics import confusion_matrix

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def generate_eda_images():
    """Genera imágenes del análisis exploratorio de datos."""
    print("="*60)
    print(" GENERANDO IMÁGENES DE EDA")
    print("="*60)
    
    # Crear carpeta images si no existe
    os.makedirs("images", exist_ok=True)
    
    # Cargar datos limpios
    cleaner = TitanicCleaner("data/titanic.csv")
    cleaner.load_data()
    cleaner.clean_data()
    cleaner.feature_engineering()
    df = cleaner.df
    
    #Distribucion de supervivencia
    print("\nGenerando survived_distribution.png...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df, x='Survived', palette=['#e74c3c', '#2ecc71'])
    plt.title('Distribución de Supervivencia en el Titanic', fontsize=14, fontweight='bold')
    plt.xlabel('Supervivencia (0 = No, 1 = Sí)')
    plt.ylabel('Cantidad de Pasajeros')
    
    # Añadir porcentajes
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / len(df):.1f}%'
        ax.text(p.get_x() + p.get_width()/2., p.get_height() + 10, 
                percentage, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("images/survived_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" Guardado")
    
    #Variables numericas vs supervivencia
    print("Generando numeric_vs_survived.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Age vs Survived
    sns.boxplot(data=df, x='Survived', y='Age', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Edad vs Supervivencia', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Supervivencia')
    axes[0].set_ylabel('Edad')
    
    # Fare vs Survived
    sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Tarifa vs Supervivencia', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Supervivencia')
    axes[1].set_ylabel('Tarifa')
    
    plt.tight_layout()
    plt.savefig("images/numeric_vs_survived.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" Guardado")
    
    #Variables categoricas vs supervivencia
    print("Generando categorical_vs_survived.png...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Pclass vs Survived
    sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[0], palette=['#e74c3c', '#2ecc71'])
    axes[0].set_title('Clase vs Supervivencia', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Clase del Boleto')
    axes[0].legend(title='Supervivencia')
    
    # Sex vs Survived
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[1], palette=['#e74c3c', '#2ecc71'])
    axes[1].set_title('Sexo vs Supervivencia', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sexo')
    axes[1].legend(title='Supervivencia')
    
    # Embarked vs Survived
    sns.countplot(data=df, x='Embarked', hue='Survived', ax=axes[2], palette=['#e74c3c', '#2ecc71'])
    axes[2].set_title('Puerto de Embarque vs Supervivencia', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Puerto')
    axes[2].legend(title='Supervivencia')
    
    plt.tight_layout()
    plt.savefig("images/categorical_vs_survived.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardado")    
    
    #Matriz de correlacion
    print("Generando correlation_matrix.png...")
    plt.figure(figsize=(10, 8))
    
    # Seleccionar solo columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    correlation = df[numeric_cols].corr()
    
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                fmt=".2f", linewidths=0.5, square=True)
    plt.title('Matriz de Correlación (Variables Numéricas)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("images/correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(" Guardado")
    
    print("\nTodas las imágenes de EDA generadas correctamente")


def generate_model_images():
    """Genera imágenes del modelo (confusion matrix, feature importance, comparison)."""
    print("\n" + "="*60)
    print(" GENERANDO IMÁGENES DEL MODELO")
    print("="*60)
    
    # Cargar datos y entrenar modelo
    cleaner = TitanicCleaner("data/titanic.csv")
    cleaner.load_data()
    cleaner.clean_data()
    cleaner.feature_engineering()
    
    X_train, X_test, y_train, y_test = cleaner.separar_datos()
    num_cols, cat_cols = cleaner.get_feature_lists(X_train)
    
    # Entrenar modelo
    model = TitanicModel()
    model.create_preprocesor(num_cols, cat_cols)
    model.create_pipeline()
    
    param_grid = {
        'modelo__n_estimators': [100, 200, 300],
        'modelo__max_depth': [10, 20, 30, None],
        'modelo__min_samples_split': [2, 5, 10],
        'modelo__min_samples_leaf': [1, 2, 4]
    }
    best_model = model.optimize_hyperparameters(X_train, y_train, param_grid)
    
    #Matriz de confusion
    print("\n Generando confusion_matrix.png...")
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Sobrevivió', 'Sobrevivió'],
                yticklabels=['No Sobrevivió', 'Sobrevivió'])
    plt.title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Real', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardado")
    
    print("Generando feature_importance.png...")
    model.get_feature_importance(num_cols, cat_cols, top_n=10)
    
    #Comparacion de modelos
    print("Generando model_comparison.png...")
    model.compare_models(X_train, y_train, X_test, y_test, num_cols, cat_cols)
    
    print("\nTodas las imágenes del modelo generadas correctamente")


def main():    
    # Generar imágenes de EDA
    generate_eda_images()
    
    # Generar imágenes del modelo
    generate_model_images()
    
    print("\n" + "="*60)    
    print(" TODAS LAS IMÁGENES GENERADAS EXITOSAMENTE")
    print("="*60)
    print("\nImágenes disponibles en la carpeta 'images/':")
    print("  - survived_distribution.png")
    print("  - numeric_vs_survived.png")
    print("  - categorical_vs_survived.png")
    print("  - correlation_matrix.png")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - model_comparison.png")
    print("\n¡Listo para usar en tu README!")


if __name__ == "__main__":
    main()