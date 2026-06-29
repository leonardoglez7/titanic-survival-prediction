import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from .data_cleaning import TitanicCleaner
import os
import joblib
from sklearn.metrics import precision_score, recall_score, classification_report, f1_score, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicModel:
    
    def __init__(self):
        self.pipeline = None
        self.best_model = None
        self.preprocesor = None
    
    def create_preprocesor(self, num_cols, cat_cols):        
        try:
            print("="*50)
            print("Creando preprocesador")
            print("="*50)
            
            self.preprocesor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ])
            
            print("Preprocesador creado")  
        except Exception as e:
            print(f"Error en create_preprocesor: {e}")      
            raise

    def create_pipeline(self, model=RandomForestClassifier(random_state=42)):           
        try:               
               
            print("="*50)
            print("Creando pipeline")
            print("="*50)
            
            self.pipeline = Pipeline([
            ("preprocesador", self.preprocesor),
            ("modelo", model)
             ])
                
            print("Pipeline creado") 
        except Exception as e:
            print(f"Error en create_pipeline(): {e}")                                                                                            
    def train_baseline(self, X_train, y_train):
        try:
            if self.pipeline is None:
                raise ValueError("Crea el pipeline primero")
            
            value = cross_val_score(self.pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
            
            print("="*50)
            print("Validando modelo")
            print("="*50)
            
            print(f"Precision del modelo baseline: {value}")
        except Exception as e : 
            print(f"Error en train_baseline: {e}")

    def optimize_hyperparameters(self, X_train, y_train, param_grid, model_path="models/rf_model.pkl"):
        try:
            if os.path.exists(model_path):
                print("="*50)
                print(f"Modelo encontrado: {model_path}")
                print("="*50)
                
                model = self.load_model(model_path)
                return model
                
            print("="*50)
            print("Busqueda de hiperparametros")
            print("="*50)
            
            grid = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy',n_jobs=-1,verbose=2)
            
            grid.fit(X_train, y_train)
            
            self.best_model = grid.best_estimator_
            
            self.save_model(name=model_path)      
            
            print(f"️ Mejores params: {grid.best_params_}")
            return self.best_model

        except Exception as e:
            print(f"Error en optimize_hyperparametros: {e}")   
            raise                     
    
    @staticmethod
    def evaluate_model(best_model, X_test, y_test):
        try:
            print("="*50)
            print("Calculando metricas")
            print("="*50)    
    
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]
    
            metrics = {
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'confusion_matrix': confusion_matrix(y_test,y_pred),
                'accuracy':accuracy_score(y_test, y_pred)
                    }
            
            print(f"\n{'='*50}")
            print("EVALUACIÓN FINAL")
            print(f"{'='*50}")
            print(classification_report(y_test, y_pred))
    
            return metrics                         
        except Exception as e:
            print(f"Error en evaluate_model(): {e}")            

    
    def get_feature_importance(self, num_cols, cat_cols, top_n=10):
        try:
            if self.best_model is None:
                raise ValueError("Primero debes entrenar el modelo")
        
            # Extraer el modelo del pipeline
            model = self.best_model.named_steps["modelo"]
            preprocessor = self.best_model.named_steps["preprocesador"]
        
            # Obtener nombres de columnas después del OneHotEncoding
            cat_encoder = preprocessor.named_transformers_["cat"]
            cat_features = cat_encoder.get_feature_names_out(cat_cols)
        
            # Combinar todas las features
            all_features = num_cols + list(cat_features)
        
            # Extraer importancias
            importances = model.feature_importances_
        
            # Crear DataFrame
            feature_df = pd.DataFrame({
                "feature": all_features,
                "importance": importances
            }).sort_values("importance", ascending=False)
        
            # Mostrar top N
            print(f"\n{'='*50}")
            print(f"Top {top_n} Variables más importantes")
            print(f"{'='*50}")
            print(feature_df.head(top_n))
        
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x="importance", 
                y="feature", 
                data=feature_df.head(top_n),
                palette="viridis"
            )
            plt.title("Importancia de Características - Titanic", fontsize=14, fontweight='bold')
            plt.xlabel("Importancia", fontsize=12)
            plt.ylabel("Característica", fontsize=12)
            plt.tight_layout()
        
            # Guardar imagen
            os.makedirs("images", exist_ok=True)
            plt.savefig("images/feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()
        
            print(f" Gráfico guardado en: images/feature_importance.png")
        
            return feature_df
        
        except Exception as e:
            print(f"Error en get_feature_importance(): {e}")
            raise
    
    def compare_models(self, X_train, y_train, X_test, y_test, num_cols, cat_cols):
        try:        
            print("="*50)
            print("COMPARACIÓN DE MODELOS")
            print("="*50)
        
            print("\n Entrenando Random Forest...")
        
        # Crear preprocesador y pipeline RF
            self.create_preprocesor(num_cols, cat_cols)
            self.create_pipeline()  # Esto crea el pipeline con RandomForestClassifier
        
        # Optimizar RF
            param_grid_rf = {
            'modelo__n_estimators': [100, 200, 300],
            'modelo__max_depth': [10, 20, 30, None],
            'modelo__min_samples_split': [2, 5, 10],
            'modelo__min_samples_leaf': [1, 2, 4]
        }
        
            best_rf = self.optimize_hyperparameters(X_train, y_train, param_grid_rf)
            metrics_rf = self.evaluate_model(best_rf, X_test, y_test)
        
            print(f" Random Forest - Accuracy: {metrics_rf['accuracy']:.4f}")        
        
            print("\n Entrenando Gradient Boosting...")
        
        # Crear nuevo pipeline con Gradient Boosting
            self.create_pipeline(model=GradientBoostingClassifier(random_state=42))        
        
        # Optimizar GB
        
            param_grid_gb = {
            'modelo__n_estimators': [100, 200, 300],
            'modelo__learning_rate': [0.01, 0.1, 0.2],
            'modelo__max_depth': [3, 5, 7],
            'modelo__min_samples_split': [2, 5, 10],
            'modelo__min_samples_leaf': [1, 2, 4]
            }      
                                 
            best_gb = self.optimize_hyperparameters(
            X_train, y_train, param_grid_gb,
            model_path="/storage/emulated/0/Proyectos/Data Science/proyectos/titanic-survival-prediction/models/gb_model.pkl"
            )
            metrics_gb = self.evaluate_model(best_gb, X_test, y_test)
              
        
            print(f" Gradient Boosting - Accuracy: {metrics_gb['accuracy']:.4f}")
        
            #Comparar metricas
            print("\n" + "="*50)
            print(" TABLA COMPARATIVA")
            print("="*50)
        
            # Extraer solo las métricas numéricas (excluir confusion_matrix)
            metrics_to_compare = {
            'Random Forest': {k: v for k, v in metrics_rf.items() if k != 'confusion_matrix'},
            'Gradient Boosting': {k: v for k, v in metrics_gb.items() if k != 'confusion_matrix'}
            }
        
            df_compare = pd.DataFrame(metrics_to_compare).T
            print(df_compare.round(4))        
        
           #Determinar mejor modelo
            if metrics_rf['accuracy'] > metrics_gb['accuracy']:
                print(f"\n GANADOR: Random Forest (Accuracy: {metrics_rf['accuracy']:.4f})")
                self.best_model = best_rf
                self.save_model(name="models/best_model.pkl")
                return best_rf
            else:
                print(f"\nGANADOR: Gradient Boosting (Accuracy: {metrics_gb['accuracy']:.4f})")
                self.best_model = best_gb
                self.save_model(name="models/best_model.pkl")
                return best_gb
        
        except Exception as e:
            print(f" Error en compare_models(): {e}")
            raise
                                      
    def save_model(self, name="models/best_model.pkl"):
        try:
            if self.best_model is not None:
                print("="*50)
                print("Guardando modelo")
                print("="*50)
                joblib.dump(self.best_model, name)
        except Exception as e:
            print(f"Error en save_model(): {e}")
            
    def load_model(self, model_path):
        try:
            print("="*50)
            print("Cargando modelo")
            print("="*50)
            self.best_model = joblib.load(model_path)
            return self.best_model
        except Exception as e:
            print(f"Error en load_model(): {e}")              
                                                          

