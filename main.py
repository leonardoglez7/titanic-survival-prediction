from src.data_cleaning import TitanicCleaner
from src.model_pipeline import TitanicModel
import os

def main():
    print("="*60)
    print(" TITANIC SURVIVAL PREDICTION")
    print("="*60)
    
    #Limpieza de datos
    print("\n  Carga y Limpieza de Datos")
    print("-"*60)
    
    cleaner = TitanicCleaner("data/titanic.csv")
    cleaner.load_data()
    cleaner.data_analysis()
    cleaner.clean_data()
    cleaner.feature_engineering()
    cleaner.save_data("data/Titanic_clean.csv")
    
    #Separando datos
    print("\nSeparación de Datos")
    print("-"*60)
    
    X_train, X_test, y_train, y_test = cleaner.separar_datos()
    num_cols, cat_cols = cleaner.get_feature_lists(X_train)
    
    print(f"Columnas numéricas: {num_cols}")
    print(f"Columnas categóricas: {cat_cols}")
    
    #Entrenando modelo baseline
    print("\n Entrenamiento de Modelos")
    print("-"*60)
    
    titanic_model = TitanicModel()
    titanic_model.create_preprocesor(num_cols, cat_cols)
    titanic_model.create_pipeline()
    titanic_model.train_baseline(X_train, y_train)
    
    #Optimizar Random Forest
    print("\n Optimización de Random Forest")
    print("-"*60)
    
    param_grid_rf = {
        'modelo__n_estimators': [100, 200, 300],
        'modelo__max_depth': [10, 20, 30, None],
        'modelo__min_samples_split': [2, 5, 10],
        'modelo__min_samples_leaf': [1, 2, 4]
    }
    
    best_rf = titanic_model.optimize_hyperparameters(
        X_train, y_train, param_grid_rf,
        model_path="models/rf_model.pkl"
    )
    
    metrics_rf = titanic_model.evaluate_model(best_rf, X_test, y_test)
    
    #Comparar modelos
    print("\n Comparación de Modelos")
    print("-"*60)
    
    best_model = titanic_model.compare_models(
        X_train, y_train, X_test, y_test, num_cols, cat_cols
    )
    
    #Importancia de features
    print("\n FASE 6: Importancia de Variables")
    print("-"*60)
    
    titanic_model.get_feature_importance(num_cols, cat_cols, top_n=10)
    
    
    print("\n" + "="*60)
    print("PROYECTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"Mejor modelo guardado en: models/best_model.pkl")
    print(f"Imágenes generadas en: images/")
    print(f"Dataset limpio en: data/Titanic_clean.csv")

if __name__ == "__main__":
    main()