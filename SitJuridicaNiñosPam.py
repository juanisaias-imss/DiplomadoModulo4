import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# ENCABEZADO
# --------------------------------------------------

st.title("Modelo de Predicción de la Situación Jurídica de Menores")

st.image(
    "children-playing-group.jpg",
    caption="Clasificación de la situación jurídica mediante aprendizaje supervisado"
)

# --------------------------------------------------
# CARGA DE DATOS
# --------------------------------------------------

datos = pd.read_csv(
    "datos_dif2.csv",
    encoding="latin-1"
)

# --------------------------------------------------
# TÍTULO PRINCIPAL
# --------------------------------------------------

st.header(
    "Predicción del estatus jurídico de niñas, niños y adolescentes"
)

# --------------------------------------------------
# FORMULARIO DE CAPTURA
# --------------------------------------------------

def user_input_features():

    st.sidebar.header("Captura de Información")

    with st.sidebar.form("formulario_prediccion"):

        Area = st.selectbox(
            "Área",
            sorted(datos["Area"].dropna().unique())
        )

        MUNICPIO_C_I = st.selectbox(
            "Municipio",
            sorted(datos["MUNICPIO_C_I"].dropna().unique())
        )

        GENERO = st.radio(
            "Género",
            sorted(datos["GENERO"].dropna().unique())
        )

        edad_meses = st.slider(
            "Edad (meses)",
            min_value=int(datos["edad_meses"].min()),
            max_value=int(datos["edad_meses"].max()),
            value=int(datos["edad_meses"].median())
        )

        antiguedad_caso = st.slider(
            "Antigüedad del caso",
            min_value=int(datos["antiguedad del caso"].min()),
            max_value=int(datos["antiguedad del caso"].max()),
            value=int(datos["antiguedad del caso"].median())
        )

        Abogado = st.selectbox(
            "Abogado",
            sorted(datos["Abogado"].dropna().unique())
        )

        submitted = st.form_submit_button(
            "Generar Predicción"
        )

    data = {
        "Area": Area,
        "edad_meses": edad_meses,
        "antiguedad del caso": antiguedad_caso,
        "MUNICPIO_C_I": MUNICPIO_C_I,
        "GENERO": GENERO,
        "Abogado": Abogado
    }

    return pd.DataFrame(data, index=[0]), submitted


df, submitted = user_input_features()

# --------------------------------------------------
# MOSTRAR DATOS CAPTURADOS
# --------------------------------------------------

st.subheader("Datos Capturados")
st.dataframe(df)

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
# CODIFICACIÓN VARIABLE OBJETIVO
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
# ESCALAMIENTO DE VARIABLES NUMÉRICAS
# --------------------------------------------------

variables_numericas = [
    "edad_meses",
    "antiguedad del caso"
]

# Convertir a float antes del escalamiento
X_train[variables_numericas] = (
    X_train[variables_numericas].astype(float)
)

X_test[variables_numericas] = (
    X_test[variables_numericas].astype(float)
)

df_encoded[variables_numericas] = (
    df_encoded[variables_numericas].astype(float)
)

# Crear escalador
scaler = StandardScaler()

# Ajustar con entrenamiento
X_train[variables_numericas] = scaler.fit_transform(
    X_train[variables_numericas]
)

# Transformar prueba
X_test[variables_numericas] = scaler.transform(
    X_test[variables_numericas]
)

# Transformar datos capturados por el usuario
df_encoded[variables_numericas] = scaler.transform(
    df_encoded[variables_numericas]
)

# --------------------------------------------------
# MODELO RANDOM FOREST
# --------------------------------------------------
modelo = RandomForestClassifier(

    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)



# --------------------------------------------------
# ENTRENAMIENTO
# --------------------------------------------------

modelo.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# EVALUACIÓN
# --------------------------------------------------

y_pred = modelo.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

# --------------------------------------------------
# MÉTRICAS DEL MODELO
# --------------------------------------------------

st.sidebar.subheader("Desempeño del Modelo")

st.sidebar.write(
    f"Exactitud: {accuracy:.2%}"
)

st.sidebar.write(
    f"Precisión: {precision:.2%}"
)

st.sidebar.write(
    f"Recall: {recall:.2%}"
)

st.sidebar.write(
    f"F1 Score: {f1:.2%}"
)

# --------------------------------------------------
# PREDICCIÓN
# --------------------------------------------------

if submitted:

    prediction = modelo.predict(
        df_encoded
    )

    estatus_predicho = le_target.inverse_transform(
        prediction
    )

    st.subheader(
        "Resultado de la Predicción"
    )

    st.success(
        f"✅ Estatus predicho: {estatus_predicho[0]}"
    )

    probabilidades = modelo.predict_proba(
        df_encoded
    )[0]

    clases = le_target.classes_

    prob_df = pd.DataFrame({
        "Estatus": clases,
        "Probabilidad": probabilidades
    })

    st.subheader(
        "Probabilidades por Estatus"
    )

    st.dataframe(
        prob_df.sort_values(
            by="Probabilidad",
            ascending=False
        ).reset_index(drop=True)
    )
