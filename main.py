import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Importando SMOTE
import joblib  # Importando joblib

# Carregar os dados
data0 = pd.read_csv("/content/drive/MyDrive/colab/BaseCovid/lbp-train-fold_0.csv")
data1 = pd.read_csv("/content/drive/MyDrive/colab/BaseCovid/lbp-train-fold_1.csv")
data2 = pd.read_csv("/content/drive/MyDrive/colab/BaseCovid/lbp-train-fold_2.csv")
data3 = pd.read_csv("/content/drive/MyDrive/colab/BaseCovid/lbp-train-fold_3.csv")
data4 = pd.read_csv("/content/drive/MyDrive/colab/BaseCovid/lbp-train-fold_4.csv")

# Concatenar os dados
df = pd.concat([data0, data1, data2, data3, data4], ignore_index=True)

# Verificar e remover valores nulos
df = df.dropna()

# Separar a coluna 'class' (alvo) e as características
X = df.select_dtypes(include=[np.number]).drop(columns=['class'], errors='ignore')
y = df['class']

# Escalar os dados
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Aplicar SMOTE para balancear o conjunto de dados
smote = SMOTE(random_state=42, k_neighbors=2)  # Ajustando o número de vizinhos
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Verificando as formas após o balanceamento
print("X_train_balanced shape:", X_train_balanced.shape)
print("y_train_balanced shape:", y_train_balanced.shape)

# Treinar o modelo com o conjunto de treino balanceado
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train_balanced, y_train_balanced)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia: {accuracy:.2f}")

# Gerar relatório de classificação
report = classification_report(y_test, y_pred)
print("Relatório de Classificação:")
print(report)

# Calcular e exibir a matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# Salvar o modelo treinado
model_filename = '/content/drive/MyDrive/colab/BaseCovid/random_forest_model.joblib'
joblib.dump(clf, model_filename)
print(f"Modelo salvo em: {model_filename}")