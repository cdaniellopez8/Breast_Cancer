import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import requests
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns


  
# Cargar el dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# Unir todo en un dataframe 
df = pd.DataFrame(X, columns=breast_cancer_wisconsin_diagnostic.feature_names)
df['Diagnóstico'] = y
df['Diagnóstico'] = df['Diagnóstico'].map({"M": 'Maligno', "B": 'Benigno'})

# Sidebar

# Autor

st.sidebar.header('Variables de estudio')
variables = df.columns
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

# Tabla de datos
st.subheader("Vista previa de los datos")
st.dataframe(df[[variable_seleccionada, otra_variable, 'Diagnóstico']].head(6))

# Correlacion

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



