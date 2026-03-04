# model.py
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, recall_score,
    precision_score, roc_auc_score
)


def entrenar_modelo(df):

    X = df[["Temperatura", "Precipitacion",
            "Humedad", "Indice_socioeconomico"]]
    y = df["Presencia_Aedes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }

    return modelo, metricas


def calcular_probabilidades(modelo, df):
    X = df[["Temperatura", "Precipitacion",
            "Humedad", "Indice_socioeconomico"]]

    df["Probabilidad"] = modelo.predict_proba(X)[:, 1]
    return df


