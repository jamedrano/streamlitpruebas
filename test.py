import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from xgboost import XGBRegressor
import sklearn.metrics as mt
import pickle
import xgboost as xgb

@st.cache_data
def cargar_datos(archivo_subido, hoja, encabezado):
    try:
        datos = pd.read_excel(archivo_subido, header=encabezado, sheet_name=hoja, engine='openpyxl')
        datos.columns = datos.columns.str.strip()
        for col in datos.columns:
            if datos[col].dtype == 'O':
                datos[col] = datos[col].str.strip()
        return datos
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return None

@st.cache_resource
def cargar_modelo(archivo_subido):
    try:
        modelo = pd.read_pickle(archivo_subido)
        return modelo
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def pegar(df1, df2):
    return pd.concat([df1, df2.set_index(df1.index)], axis=1)

def a_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'})
        worksheet.set_column('A:A', None, format1)
    processed_data = output.getvalue()
    return processed_data

def entrenar_modelo(datos, columnas_a_quitar, respuesta):
    eta = 0.08
    reg_lambda = 5
    X = datos.drop(columnas_a_quitar, axis=1)
    y = datos[respuesta]
    modeloXGB = XGBRegressor(booster='gblinear', eta=eta, reg_lambda=reg_lambda)
    modeloXGB.fit(X, y)
    predicciones = modeloXGB.predict(X)

    st.download_button("Descargar Modelo", data=pickle.dumps(modeloXGB), file_name="model.pkl")
    return X, y, predicciones

def desplegar_resultados(subdatos2, columnas_a_quitar, respuesta):
    X, y, predicciones = entrenar_modelo(subdatos2, columnas_a_quitar, respuesta)
    subset1 = subdatos2.drop(columnas_a_quitar, axis=1)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)
    ax.scatter(y, predicciones)
    st.pyplot(fig)
    
    st.write("Porcentaje de Error")
    st.write(mt.mean_absolute_percentage_error(y, predicciones))
    st.write("Coeficiente de Determinación")
    st.write(mt.r2_score(y, predicciones))
    
    datos_resultados = pd.DataFrame({'y_real': y, 'predicciones': predicciones})
    subset2 = pegar(subset1, datos_resultados)
    st.dataframe(subset2)
    
    df_xlsx = a_excel(subset2)
    st.download_button(label='📥 Descargar datos', data=df_xlsx, file_name='df_test.xlsx')

st.set_page_config(page_title='Modelo Predictivo Resistencia a la Compresión CEMPRO', layout="wide")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Datos', 'Descripción de Datos', 'Gráficos', 'Entrenar Modelo', 'Aplicar Modelo'])

st.sidebar.write("****Cargar Archivo de Datos en Excel****")
archivo_subido = st.sidebar.file_uploader("*Subir Archivo Aquí*")

if archivo_subido:
    hoja = st.sidebar.selectbox("*¿Qué hoja contiene los datos?*", pd.ExcelFile(archivo_subido).sheet_names)
    encabezado = st.sidebar.number_input("*¿Qué fila contiene los nombres de columnas?*", min_value=0, max_value=100)
    
    datos = cargar_datos(archivo_subido, hoja, encabezado)
    
    if datos is not None:
        with tab1:
            st.write('### 1. Datos Cargados')
            st.dataframe(datos, use_container_width=True)

        with tab2:
            st.write('### 2. Descripción de los Datos')
            opcion_descripcion = st.radio("**¿Qué desea ver de los datos?**", 
                                          ["Dimensiones", "Descripción de las Variables", "Estadísticas Descriptivas", "Conteo de Valores por Columna"])
            
            if opcion_descripcion == 'Descripción de las Variables':
                descripcion_variables = datos.dtypes.reset_index().rename(columns={'index': 'Nombre del Campo', 0: 'Tipo de Dato'})
                st.dataframe(descripcion_variables, use_container_width=True)
            
            elif opcion_descripcion == 'Estadísticas Descriptivas':
                estadisticas_descriptivas = datos.describe(include='all').round(2).fillna('')
                st.dataframe(estadisticas_descriptivas, use_container_width=True)
            
            elif opcion_descripcion == 'Conteo de Valores por Columna':
                columna_a_investigar = st.selectbox("Seleccione Columna a Investigar", datos.select_dtypes('object').columns)
                conteo_valores = datos[columna_a_investigar].value_counts().reset_index().rename(columns={'index': 'Valor', columna_a_investigar: 'Conteo'})
                st.dataframe(conteo_valores, use_container_width=True)
            
            else:
                st.write('###### Dimensiones de los Datos:', datos.shape)

        with tab3:
            molino = st.selectbox("**Seleccione Molino**", datos['Molino'].unique())
            tipo_cemento = st.selectbox("**Seleccione Tipo de Cemento**", datos['Tipo de Cemento'].unique())
            tipo_grafico = st.selectbox("**Seleccione Tipo de Gráfico**", ['Cajas', 'Histograma', 'Tendencia'])
            subdatos = datos[(datos['Tipo de Cemento'] == tipo_cemento) & (datos['Molino'] == molino)]
            
            st.write('### 3. Exploración Gráfica')
            if tipo_grafico == "Cajas":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 6)
                axs[0, 0].boxplot(subdatos['R1D'])
                axs[0, 0].set_title("1 día")
                axs[0, 1].boxplot(subdatos['R3D'])
                axs[0, 1].set_title("3 días")
                axs[1, 0].boxplot(subdatos['R7D'])
                axs[1, 0].set_title("7 días")
                axs[1, 1].boxplot(subdatos['R28D'])
                axs[1, 1].set_title("28 días")
                st.pyplot(fig)
            elif tipo_grafico == "Histograma":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 6)
                axs[0, 0].hist(subdatos['R1D'])
                axs[0, 0].set_title("1 día")
                axs[0, 1].hist(subdatos['R3D'])
                axs[0, 1].set_title("3 días")
                axs[1, 0].hist(subdatos['R7D'])
                axs[1, 0].set_title("7 días")
                axs[1, 1].hist(subdatos['R28D'])
                axs[1, 1].set_title("28 días")
                st.pyplot(fig)
            elif tipo_grafico == "Tendencia":
                fig, axs = plt.subplots(2, 2)
                fig.set_size_inches(10, 8)
                axs[0, 0].plot(subdatos['Fecha'], subdatos['R1D'])
                axs[0, 0].set_title("1 día")
                axs[0, 0].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[0, 1].plot(subdatos['Fecha'], subdatos['R3D'])
                axs[0, 1].set_title("3 días")
                axs[0, 1].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[1, 0].plot(subdatos['Fecha'], subdatos['R7D'])
                axs[1, 0].set_title("7 días")
                axs[1, 0].tick_params(axis='x', labelrotation=30, labelsize=8)
                axs[1, 1].plot(subdatos['Fecha'], subdatos['R28D'])
                axs[1, 1].set_title("28 días")
                axs[1, 1].tick_params(axis='x', labelrotation=30, labelsize=8)
                st.pyplot(fig)

        with tab4:
            molino2 = st.selectbox("**Seleccione Molino a Modelar**", datos['Molino'].unique())
            tipo_cemento2 = st.selectbox("**Seleccione Tipo de Cemento a Modelar**", datos['Tipo de Cemento'].unique())
            edad = st.selectbox("**Edad a Predecir**", ["1 día", "3 días", "7 días", "28 días"])
            
            subdatos2 = datos[(datos['Tipo de Cemento'] == tipo_cemento2) & (datos['Molino'] == molino2)]
            
            if edad == "1 día":
                columnas_a_quitar = ['Fecha', 'Tipo de Cemento', 'Molino', 'R1D', 'R3D', 'R7D', 'R28D']
                respuesta = 'R1D'
                desplegar_resultados(subdatos2, columnas_a_quitar, respuesta)
                
            elif edad == "3 días":
                columnas_a_quitar = ['Fecha', 'Tipo de Cemento', 'Molino', 'R3D', 'R7D', 'R28D']
                respuesta = 'R3D'
                desplegar_resultados(subdatos2, columnas_a_quitar, respuesta)
                
            elif edad == "7 días":
                columnas_a_quitar = ['Fecha', 'Tipo de Cemento', 'Molino', 'R7D', 'R28D']
                respuesta = 'R7D'
                desplegar_resultados(subdatos2, columnas_a_quitar, respuesta)
                
            elif edad == "28 días":
                columnas_a_quitar = ['Fecha', 'Tipo de Cemento', 'Molino', 'R28D']
                respuesta = 'R28D'
                desplegar_resultados(subdatos2, columnas_a_quitar, respuesta)

        with tab5:
            archivo_modelo = st.file_uploader("Cargar Modelo")
            if archivo_modelo:
                modelo_prod = cargar_modelo(archivo_modelo)
                if modelo_prod:
                    st.write("Modelo cargado con éxito")
                    datos_prod_archivo = st.file_uploader("Cargar Datos de Producción")
                    if datos_prod_archivo:
                        datos_prod = cargar_datos(datos_prod_archivo, 'Sheet1', 0)
                        if datos_prod is not None:
                            st.write("Realizando predicciones...")
                            y_pred = modelo_prod.get_booster().predict(xgb.DMatrix(datos_prod))
                            predicciones_df = pd.DataFrame({'Predicciones': y_pred})
                            resultados = pegar(datos_prod, predicciones_df)
                            st.dataframe(resultados)
                            
                            datos_excel = a_excel(resultados)
                            st.download_button(label='📥 Descargar resultados', data=datos_excel, file_name='resultados.xlsx')
