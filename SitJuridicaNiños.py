
import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.write(
    """
    # Predicción de la situación jurídica de menores en condición de vulnerabilidad
    """
)

st.image(
    "children-817368_1280.jpg",
    caption="Clasificando la situación jurídica del niño mediante un modelo de aprendizaje supervisado"
)

# --------------------------------------------------
# CARGA DE DATOS PARA CATÁLOGOS
# --------------------------------------------------

datos = pd.read_csv("datos_dif2.csv", encoding="latin-1")

# --------------------------------------------------
# TÍTULO
# --------------------------------------------------

st.title("Modelo de Predicción de Situación Jurídica")

st.write(
    "Predicción del estatus jurídico de niñas, niños y adolescentes"
)

# --------------------------------------------------
# CAPTURA DE DATOS
# --------------------------------------------------

def user_input_features():

    Area = st.selectbox(
        "Área",
        sorted(datos["Area"].dropna().unique())
    )

    edad_meses = st.number_input(
        "Edad (meses)",
        min_value=float(datos["edad_meses"].min()),
        max_value=float(datos["edad_meses"].max()),
        value=float(datos["edad_meses"].median())
    )

    antiguedad_caso = st.number_input(
        "Antigüedad del caso",
        min_value=float(datos["antiguedad del caso"].min()),
        max_value=float(datos["antiguedad del caso"].max()),
        value=float(datos["antiguedad del caso"].median())
    )

    MUNICPIO_C_I = st.selectbox(
        "Municipio",
        sorted(datos["MUNICPIO_C_I"].dropna().unique())
    )

    GENERO = st.selectbox(
        "Género",
        sorted(datos["GENERO"].dropna().unique())
    )

    Abogado = st.selectbox(
        "Abogado",
        sorted(datos["Abogado"].dropna().unique())
    )

    data = {
        "Area": Area,
        "edad_meses": edad_meses,
        "antiguedad del caso": antiguedad_caso,
        "MUNICPIO_C_I": MUNICPIO_C_I,
        "GENERO": GENERO,
        "Abogado": Abogado
    }

    return pd.DataFrame(data, index=[0])


df = user_input_features()

st.subheader("Datos capturados")

st.write(df)

# --------------------------------------------------
# DATOS DE ENTRENAMIENTO
# --------------------------------------------------

datos_entrenamiento = pd.read_csv(
    "datos_dif2.csv",
    encoding="latin-1"
)

# --------------------------------------------------
# VARIABLES PREDICTORAS Y OBJETIVO
# --------------------------------------------------

X = datos_entrenamiento[
    [
        "Area",
        "edad_meses",
        "antiguedad del caso",
        "MUNICPIO_C_I",
        "GENERO",
        "Abogado"
    ]
]

y = datos_entrenamiento["Estatus"]

# --------------------------------------------------
# CODIFICACIÓN DE VARIABLE OBJETIVO
# --------------------------------------------------

le_target = LabelEncoder()

y_encoded = le_target.fit_transform(y)

# --------------------------------------------------
# ONE HOT ENCODING
# --------------------------------------------------

X_encoded = pd.get_dummies(X)

df_encoded = pd.get_dummies(df)

df_encoded = df_encoded.reindex(
    columns=X_encoded.columns,
    fill_value=0
)

# --------------------------------------------------
# DIVISIÓN DE DATOS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# --------------------------------------------------
# MODELO RANDOM FOREST
# --------------------------------------------------

modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42
)

# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

modelo.fit(X_train, y_train)

# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------

prediction = modelo.predict(
    df_encoded
)

estatus_predicho = le_target.inverse_transform(
    prediction
)

st.subheader("Predicción")

st.success(
    f"Estatus predicho: {estatus_predicho[0]}"
)
