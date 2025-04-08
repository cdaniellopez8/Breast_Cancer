import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import requests
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


  
# Cargar el dataset
#breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
#X = breast_cancer_wisconsin_diagnostic.data.features 
#y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# Unir todo en un dataframe 
#df = pd.DataFrame(X, columns=breast_cancer_wisconsin_diagnostic.feature_names)
#df['Diagnóstico'] = y
#df['Diagnóstico'] = df['Diagnóstico'].map({"M": 'Maligno', "B": 'Benigno'})



datos = pd.read_csv("data.csv")

X = datos.drop(columns=["id", "diagnosis"])
feature_names = X.columns.tolist()
y = datos["diagnosis"]
df = pd.DataFrame(X, columns=feature_names)
df["Diagnóstico"] = y.map({"M": "Maligno", "B": "Benigno"})

# Sidebar

# Autor

st.sidebar.header('Variables de estudio')
variables = df.columns.drop("Diagnóstico")
variable_seleccionada = st.sidebar.selectbox('Por favor, seleccione la variable de interés', variables)

with st.sidebar:
    st.markdown("""
    <hr>
    <div style="text-align: center; font-size: 0.9em; color: gray;">
        Desarrollado por Carlos D. López P.
    </div>
    """, unsafe_allow_html=True)

# Título y descripción
st.title("Análisis Exploratorio de Datos - Breast Cancer (Wisconsin)")
st.markdown("""
    <div style="text-align: justify;">
        Este análisis permite explorar las características más relevantes del dataset <strong>Breast Cancer Wisconsin Diagnostic</strong>, 
        proporcionando visualizaciones y estadísticas descriptivas para facilitar la comprensión del comportamiento de cada variable 
        según el diagnóstico.
    </div>
    """, unsafe_allow_html=True)


st.subheader("Primeras filas del dataset")
st.dataframe(df.head(6))
st.markdown("""
    <div style="text-align: justify;">
        Tras llevar a cabo un análisis exploratorio inicial del conjunto de datos, se verificó la ausencia de valores faltantes en las variables consideradas. 
    </div>
    """, unsafe_allow_html=True)


#st.subheader("Conteo de datos faltantes en el dataset")

#valores_nulos = df.isnull().sum()

#tabla_nulos = pd.DataFrame({
#    'Variable': valores_nulos.index,
#    'Cantidad de valores faltantes': valores_nulos.values
#})

#st.markdown("""
#    <div style="text-align: justify;">
#        Como se evidencia en la presente tabla, vemos que no hay valores faltantes en el dataset.
#    </div>
#    """, unsafe_allow_html=True)

#st.table(tabla_nulos)




st.markdown("""
---
""")

# Título variable seleccionada
st.markdown(f"### Análisis de la variable: <span style='color:#2a9df4; font-weight:bold'>{variable_seleccionada}</span>", unsafe_allow_html=True)

# Subconjunto de datos
valores = df[variable_seleccionada]
diagnostico = df['Diagnóstico']

# Gráficos
st.subheader("Distribución de la variable")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.histplot(valores, kde=True, bins=30, color='steelblue', ax=ax)
    ax.set_title("Histograma")
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=diagnostico, y=valores, palette='Set2', ax=ax2)
    ax2.set_title("Boxplot por Diagnóstico")
    st.pyplot(fig2)


# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.markdown("""
    <div style="overflow-x: auto; max-width: 100%;">
""", unsafe_allow_html=True)

st.dataframe(valores.describe().to_frame().T.round(2))

st.markdown("</div>", unsafe_allow_html=True)

# Comparación por diagnóstico
st.markdown(f"### Resumen de <span style='color:#2a9df4; font-weight:bold'>{variable_seleccionada}</span> por tipo de diagnóstico", unsafe_allow_html=True)
st.write(df.groupby("Diagnóstico")[variable_seleccionada].describe())


st.markdown("""
---
""")

# Gráfico de dispersión con otras variables
st.subheader("Relación con otras variables")
otras_variables = [var for var in variables if var != variable_seleccionada]
otra_variable = st.selectbox("Seleccione otra variable para comparar", otras_variables)

fig3, ax3 = plt.subplots()
sns.scatterplot(x=df[variable_seleccionada], y=df[otra_variable], hue=df['Diagnóstico'], palette='Set1', ax=ax3)
ax3.set_xlabel(variable_seleccionada)
ax3.set_ylabel(otra_variable)
ax3.set_title("Gráfico de dispersión")
st.pyplot(fig3)


st.markdown("""
---
""")

# Tabla de datos
st.subheader("Vista previa de los datos")
st.dataframe(df[[variable_seleccionada, otra_variable, 'Diagnóstico']].head(6))

# Correlacion


st.markdown("""
---
""")

st.subheader("Matriz de correlacion entre las variables de estudio")
st.markdown("""
    <div style="text-align: justify;">
        A continuación, se presenta la matriz de correlación, la cual permite identificar la intensidad y dirección de las relaciones entre las variables numéricas del dataset. Este análisis resulta útil para detectar posibles asociaciones relevantes que podrían influir en el modelado posterior.
    </div>
    """, unsafe_allow_html=True)

df_temp = df.copy()

# Codifica la variable 'Diagnóstico' como 0 y 1
df_temp['Diagnóstico'] = df_temp['Diagnóstico'].map({'Benigno': 0, 'Maligno': 1})

# Selecciona solo las columnas numéricas
df_numericas = df_temp.select_dtypes(include=[float, int])

# Calcula la matriz de correlación
matriz_correlacion = df_numericas.corr()

# Visualiza la matriz de correlación con un mapa de calor
correlacion_fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación')
plt.tight_layout()

st.pyplot(correlacion_fig)

st.markdown("""
## 🔍 Análisis de Correlaciones

El siguiente mapa de calor representa las correlaciones entre las variables del conjunto de datos, incluyendo la variable objetivo **`Diagnóstico`**.

### 📌 Conclusiones principales

---

### 🔹 1. Fuertes correlaciones entre ciertas variables
- Variables como `radius_mean`, `perimeter_mean` y `area_mean` muestran una **fuerte correlación positiva** entre sí.
- Este patrón también se repite en sus versiones `_worst`: `radius_worst`, `perimeter_worst`, `area_worst`.
- Es esperable, ya que están relacionadas geométricamente: un mayor radio implica mayor perímetro y mayor área.

---

### 🔹 2. Posible multicolinealidad
- Se observan correlaciones altas entre variables similares medidas en distintas etapas (por ejemplo, `radius_mean`, `radius_se`, `radius_worst`).
- Esta redundancia sugiere **potencial multicolinealidad**, que puede afectar negativamente a algunos modelos como la regresión logística.
- Técnicas como **PCA (Análisis de Componentes Principales)** o métodos de selección de variables pueden ayudar a mitigar este problema.

---

### 🔹 3. Correlación con el `Diagnóstico`
- Algunas variables tienen una **correlación positiva clara con el diagnóstico maligno**:
  - `concave points_mean`, `concavity_mean`, `radius_mean`, `perimeter_mean`, `area_mean`
  - También sus equivalentes `_worst` y algunas `_se`.
- Esto sugiere que a medida que aumentan estos valores, **es más probable que el diagnóstico sea maligno**.
- Estas variables son buenas candidatas para modelos de clasificación.

---

### 🔹 4. Variables menos correlacionadas
- Variables como `fractal_dimension_mean`, `fractal_dimension_se`, y `symmetry_mean` presentan **baja o nula correlación con el diagnóstico**.
- Aunque esto puede sugerir menor relevancia, **no se deben descartar sin una evaluación más profunda**, ya que algunas variables pueden tener relaciones no lineales con el diagnóstico.

---
""")

st.title("🧮 Prediccion del tipo de tumor")

st.markdown("""
<div style="text-align: justify;">
A continuación, puedes seleccionar un conjunto de variables para construir un modelo de regresión logística, por defecto se seleccionara la media del area, perimetro, concavidad y radio pero puedes eliminarlas o seleccionar mas variables. Una vez entrenado, podrás realizar predicciones de diagnóstico sobre nuevos datos ingresados manualmente.
</div>
""", unsafe_allow_html=True)


variables_por_defecto = ["radius_mean", "perimeter_mean", "area_mean", "concavity_mean"]

# Mostrar multiselect con preselección
variables_predictoras = st.multiselect(
    "",
    df.columns.drop("Diagnóstico"),
    default=[var for var in variables_por_defecto if var in df.columns]
)


# Selección de variables predictoras
# variables_predictoras = st.multiselect("Selecciona las variables para el modelo", df.columns.drop("Diagnóstico"))

if len(variables_predictoras) > 0:
    # División de los datos
    X = df[variables_predictoras]
    y = df['Diagnóstico'].map({"Benigno": 0, "Maligno": 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    # Evaluación del modelo
    st.subheader("Reporte del Modelo")
    y_pred = modelo.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    # Predicción individual
    st.markdown("""
    ---
    ### 🧪 Valores para Predicción
    Ingresa los valores para cada variable seleccionada:
    """)

    input_data = {}
    for var in variables_predictoras:
        input_data[var] = st.number_input(f"{var}", float(0), float(10000), float(df[var].mean()))

    if st.button("Predecir Diagnóstico"):
        input_df = pd.DataFrame([input_data])
        prediccion = modelo.predict(input_df)[0]
        probabilidad = modelo.predict_proba(input_df)[0][1]

        diagnostico = "Maligno" if prediccion == 1 else "Benigno"

        if diagnostico == "Maligno":
            st.markdown(f"<span style='color:red; font-weight:bold;'>✅ Diagnóstico predicho: {diagnostico} 🔴</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:red;'>🔬 Probabilidad de ser maligno: {probabilidad:.2%}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:green; font-weight:bold;'>✅ Diagnóstico predicho: {diagnostico} 🟢</span>", unsafe_allow_html=True)
            st.markdown(f"Probabilidad de ser maligno: <span style='color:red;'>🔬  {probabilidad:.2%}</span>", unsafe_allow_html=True)
else:
    st.info("Selecciona al menos una variable para entrenar el modelo.")
