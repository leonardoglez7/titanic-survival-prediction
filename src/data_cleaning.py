import pandas as pd
import os
from sklearn.model_selection import train_test_split

class TitanicCleaner:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        #Cargar dataset
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"No se encontro: {self.file_path}")
            
            file = pd.read_csv(self.file_path)
            df = pd.DataFrame(file)
            self.df = df
            print(f"Dataset cargado correctamente: {self.df.shape}")            
            
        except Exception as e:
            print(f"Error en load_data(): {e}")
            raise
      
    
    def data_analysis(self):
        #Analisis de la informacion
        print("="*50)
        print("Primer Analisis (Superficial)")
        print("="*50)

        print(f"Forma de los datos: {self.df.shape} \n")
        print(f"Tipos de datos: {self.df.dtypes} \n")
        print(f"Primeros datos: {self.df.head()} \n")

        print("="*50)
        print("Segundo Analisis (Nulos)")
        print("="*50)

        print(self.df.info())
        print(self.df.isnull().sum())
        print(self.df.describe())
    
    def clean_data(self):
         #Limpiar datset
        try:
             if self.df is None:
                 raise ValueError("Primero debes  argar los datos con load_data()")
             
              #Eliminar columnas PassengerId, Name, Ticket y Cabin       
             self.df = self.df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
             
             #Rellenar ceros de Embarked con la moda 
             self.df["Embarked"] = self.df["Embarked"].fillna(self.df["Embarked"].mode()[0])
             
             #Rellenar ceros de Age con la mediana por clases correspondientes
             median_ages = self.df.groupby("Pclass")["Age"].median()
             self.df["Age"] = self.df["Age"].fillna(self.df["Pclass"].map(median_ages))
             
             return self.df
             
        except Exception as e:
            print(f"Error en clean_data(): {e}")
            raise
          
    def feature_engineering(self):
        try:
             #Crear columna FamilySize
             self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] +1
             
             #Crear columna isAlone
             self.df["isAlone"] = (self.df["FamilySize"] ==1).astype(int)
        except Exception as e:
            print(f"Error en feature_engineering(): {e}")
            raise
    
    def separar_datos(self):
        #Separar datos y calcukar balanceo de class
        try:
            X = self.df.drop("Survived", axis=1)
            y = self.df["Survived"]
        
            X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            print("="*50)
            print("Calculando balanceo de clases")
            print("="*50)
            
            porcentajes = y.value_counts(normalize=True)  * 100    
            print(porcentajes)
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error en separar_datos(): {e}")
            raise
    
    def get_feature_lists(self, X_train):
        #Separar columnas numericas y categoricas 
        try:
            #Numericas
            num_cols =X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
             #Categoricas
            cat_cols = X_train.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
             
            return num_cols, cat_cols
        
        except Exception as e:
            print(f"Error en get_feature_lists(): {e}")

    def save_data(self, path):
        #Guardar dataset
        try:
            self.df.to_csv(path, index=False)
        except Exception as e:
            print(f"Error en save_data(): {e}")
                