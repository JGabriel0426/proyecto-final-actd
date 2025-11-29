import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import ast
from tensorflow import keras
import joblib
# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Airbnb Barcelona",
    layout="wide"
)
def load_classification_model():
    try:
        model = keras.models.load_model("modelo_clasificacion.h5")
        scaler = joblib.load("scaler_clasificacion.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None

model_clf, scaler_clf = load_classification_model()
# --- TÍTULO Y DESCRIPCIÓN ---
st.title("Tablero Analítico de Airbnb - Barcelona")
st.markdown("""
Este tablero permite analizar el mercado de Airbnb en Barcelona para apoyar la toma de decisiones de anfitriones como **Carina**.
Se divide en tres secciones: Análisis de Situación Actual, Predicción de Precios y Clasificación de Propiedades.
""")

# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_clean_barcelona.csv')
        
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Eliminamos filas donde la conversión pudo haber fallado (si hay NaN)
        df = df.dropna(subset=['latitude', 'longitude'])
        
        return df
    except FileNotFoundError:
        st.error("No se encontró el archivo 'data_clean_barcelona.csv'. Por favor expórtalo desde tu notebook.")
        return None

# Función para cargar modelos (placeholder)
def load_models():
    try:
        reg_model = joblib.load('modelo_prediccion_precio.pkl')
        clf_model = joblib.load('modelo_clasificacion.pkl')
        return reg_model, clf_model
    except:
        return None, None

df = load_data()
reg_model, clf_model = load_models()

if df is not None:
    
    # --- CREACIÓN DE PESTAÑAS ---
    tab1, tab2, tab3 = st.tabs(["Situación Actual", "Modelo Predictivo", "Modelo Clasificación"])

    # =============================================================================
    # PESTAÑA 1: SITUACIÓN ACTUAL (Respuesta a Preguntas de Negocio)
    # =============================================================================
    with tab1:
        st.header("Análisis del Mercado y Competitividad")

        # --- FILTROS LATERALES (Simulación perfil de Carina) ---
        # st.sidebar.header("Filtros de Análisis")
        # barrios = ["Todos"] + sorted(df["neighbourhood_cleansed"].unique().tolist())
        # neighborhood_filter = st.sidebar.selectbox("Selecciona Barrio (Simular ubicación de Carina):", barrios)
        # room_type_filter = st.sidebar.multiselect("Tipo de Habitación:", df['room_type'].unique(), default=df['room_type'].unique())

        # # Filtrar datos
        # if neighborhood_filter == "Todos":
        #     df_filtered = df[df['room_type'].isin(room_type_filter)]
        # else:
        #     df_filtered = df[
        #         (df['neighbourhood_cleansed'] == neighborhood_filter) &
        #         (df['room_type'].isin(room_type_filter))
        #     ]
        # 'property_type_clean'

        # # --- KPIs PRINCIPALES ---
        # col1, col2, col3 = st.columns(3)
        # avg_price = df_filtered['price'].mean()
        # avg_reviews = df_filtered['number_of_reviews'].mean()
        # total_listings = len(df_filtered)

        # col1.metric("Precio Promedio (Selección)", f"€ {avg_price:.2f}")
        # col2.metric("Promedio Reviews (Demanda)", f"{avg_reviews:.0f}")
        # col3.metric("Total Competidores", f"{total_listings}")

        # st.divider()

        # # --- PREGUNTA 1 & 4: Competitividad y Precio Justo (Outliers) ---
        # st.subheader("1. ¿Cómo posicionarse competitivamente y detectar precios justos?")
        # st.write("Distribución de precios en la zona seleccionada. Los puntos aislados son 'outliers' (precios atípicos).")
        
        # fig_box = px.box(
        #     df_filtered, 
        #     x="room_type", 
        #     y="price", 
        #     points="outliers",
        #     color="room_type",
        #     title=f"Distribución de Precios en {neighborhood_filter}",
        #     labels={"price": "Precio (€)", "room_type": "Tipo de Habitación"}
        # )
        # st.plotly_chart(fig_box, use_container_width=True)

        #------------------------------------------------------------------
        #
        #------------------------------------------------------------------
        # --- FILTROS LATERALES (Simulación perfil de Carina) ---
        with st.sidebar:
            st.header("Filtros de Análisis")

            df["amenities"] = df["amenities"].apply(lambda x: ast.literal_eval(x))

            # ---- Barrio ----
            barrios = ["Todos"] + sorted(df["neighbourhood_cleansed"].unique().tolist())
            neighborhood_filter = st.selectbox("Barrio:", barrios)

            # ---- Tipo de propiedad ----
            room_type_filter = st.multiselect(
                "Tipo de Propiedad:",
                df["property_type_clean"].unique(),
                default=df["property_type_clean"].unique()
            )

            # ---- n_amenities ----
            amenities_count_filter = st.multiselect(
                "Número de amenities:",
                sorted(df["n_amenities"].unique()),
                default=[]
            )

            # ---- Nuevo filtro: review_scores_rating ----
            rating_min, rating_max = st.slider(
                "Review Score Rating:",
                min_value=0.0,
                max_value=5.0,
                value=(0.0, 5.0)
            )

            # ---- Nuevo filtro: number_of_reviews ----
            rev_min, rev_max = st.slider(
                "Número de Reviews:",
                min_value=0,
                max_value=int(df["number_of_reviews"].max()),
                value=(0, int(df["number_of_reviews"].max()))
            )

        # ================== APLICAR FILTROS ==================
        df_filtered = df.copy()

        if neighborhood_filter != "Todos":
            df_filtered = df_filtered[df_filtered["neighbourhood_cleansed"] == neighborhood_filter]

        df_filtered = df_filtered[df_filtered["property_type_clean"].isin(room_type_filter)]

        if amenities_count_filter:
            df_filtered = df_filtered[df_filtered["n_amenities"].isin(amenities_count_filter)]

        df_filtered = df_filtered[
            (df_filtered["review_scores_rating"] >= rating_min) &
            (df_filtered["review_scores_rating"] <= rating_max)
        ]

        df_filtered = df_filtered[
            (df_filtered["number_of_reviews"] >= rev_min) &
            (df_filtered["number_of_reviews"] <= rev_max)
        ]

        # --- KPIs PRINCIPALES ---
        col1, col2, col3 = st.columns(3)
        avg_price = df_filtered['price'].mean()
        avg_reviews = df_filtered['number_of_reviews'].mean()
        total_listings = len(df_filtered)

        col1.metric("Precio Promedio (Selección)", f"€ {avg_price:.2f}")
        col2.metric("Promedio Reviews (Demanda)", f"{avg_reviews:.0f}")
        col3.metric("Total Competidores", f"{total_listings}")

        st.divider()
                # ============================
        #   GRILLA ESTÁTICA (NO FILTROS)
        # ============================
        st.subheader("Panorama General (Datos Completos)")

        col_g1, col_g2 = st.columns([2, 1])

        # ---- Gráfica global de distribución del precio ----
        with col_g1:
            fig_global_price = px.histogram(
                df,  # OJO → usa EL DATAFRAME COMPLETO, NO FILTRADO
                x="price",
                nbins=50,
                title="Distribución Global del Precio (Todos los Barrios)",
                opacity=0.75,
                color_discrete_sequence=["#7EA1FF"]
            )
            fig_global_price.update_layout(
                xaxis_title="Precio (€)",
                yaxis_title="Frecuencia",
                bargap=0.05
            )
            st.plotly_chart(fig_global_price, use_container_width=True)

        # ---- Calificación promedio global ----
        with col_g2:
            avg_global_rating = df["review_scores_rating"].mean()
            st.metric(
                "Calificación Promedio Global",
                f"{avg_global_rating:.2f} ",
                help="Este valor no cambia con los filtros. Es el promedio de toda la ciudad."
            )
            avg_global_occ = df["estimated_occupancy_l365d"].mean()
            st.metric(
                "Ocupación Promedio Global (%)",
                f"{avg_global_occ:.2f} ",
                help="Este valor no cambia con los filtros."
            )
            pct_superhosts = (df["host_is_superhost"].mean()) * 100
            avg_host_years = df["host_years"].mean()
            df["price_per_review"] = df["price"] / (df["number_of_reviews"] + 1)
            avg_price_per_review = df["price_per_review"].mean()
            st.metric(
                "% de Superhosts",
                f"{pct_superhosts:.2f} ",
                help="Este valor no cambia con los filtros."
            )
            st.metric(
                "Disponibilidad anual promedio",
                f"{avg_host_years:.2f} ",
                help="Este valor no cambia con los filtros."
            )
            st.metric(
                "Precio por review (eficiencia)",
                f"{avg_price_per_review:.2f} ",
                help="Este valor no cambia con los filtros."
            )

        # --- PREGUNTA 1 & 4: Competitividad y Precio Justo (Outliers) ---
        st.subheader("1. ¿Cómo posicionarse competitivamente y detectar precios justos?")
        st.write("Distribución de precios en la zona seleccionada. Los puntos aislados son 'outliers' (precios atípicos).")
        
        fig_box = px.box(
            df_filtered, 
            x="property_type_clean", 
            y="price", 
            points="outliers",
            color="property_type_clean",
            title=f"Distribución de Precios en {neighborhood_filter}",
            labels={"price": "Precio (€)", "property_type_clean": "Tipo de Habitación"}
        )
        st.plotly_chart(fig_box, use_container_width=True)

        
        # --- PREGUNTA 2: Maximizar Ingresos y Minimizar Riesgos ---
        st.subheader("2. ¿Qué propiedades maximizan ingresos?")
        st.write("Relación entre Precio y Número de Reseñas (Proxy de Ingresos/Demanda).")
        
        # Scatter plot: Precio vs Reviews
        fig_scatter = px.scatter(
            df_filtered,
            x="number_of_reviews",
            y="price",
            color="property_type_clean",
            size="review_scores_rating", # Tamaño según calificación
            hover_data=['property_type_clean', 'neighbourhood_cleansed'],
            title="Mapa de Oportunidad: Precio vs. Demanda (Reviews)",
            labels={"number_of_reviews": "Cantidad de Reseñas", "price": "Precio (€)"},
            render_mode="svg"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        
        # --- PREGUNTA 3: Zonas con Mayor Demanda ---
        st.subheader("3. ¿Qué zonas muestran mayor demanda según el barrio filtrado?")
        
        col_map, col_bar = st.columns([2, 1])
        
        # 1. Preparar datos para el mapa: usar solo el DataFrame filtrado y limpiar NaNs en coordenadas
        map_df = df_filtered.dropna(subset=['latitude', 'longitude'])
        
        with col_map:
            st.write(f"Mapa de propiedades en **{neighborhood_filter}** (Color por Precio)")
            
            if map_df.empty:
                st.warning("No hay propiedades con las coordenadas correctas para este filtro.")
            else:
                # 2. Calcular el centro dinámico del mapa
                center_lat = map_df['latitude'].mean()
                center_lon = map_df['longitude'].mean()
                
                # 3. Crear el mapa usando el DataFrame FILTRADO
                fig_map = px.scatter_mapbox(
                    map_df, # ⬅️ USAMOS df_filtered (ahora map_df)
                    lat="latitude",
                    lon="longitude",
                    color="price",
                    size="number_of_reviews",
                    color_continuous_scale=px.colors.cyclical.IceFire,
                    size_max=15,
                    zoom=13, # Aumentamos el zoom para ver un barrio
                    mapbox_style="carto-positron", # Puedes usar "open-street-map" si sigue fallando
                    title=f"Propiedades en {neighborhood_filter}"
                )
                
                # 4. Ajustar el centro del mapa
                fig_map.update_layout(
                    mapbox_center={"lat": center_lat, "lon": center_lon},
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                
                st.plotly_chart(fig_map, use_container_width=True)

        with col_bar:
            st.write("Top Barrios con más Reviews")

            # Selector del número de barrios en el TOP
            top_n = st.slider(
                "Número de barrios a mostrar (TOP N):",
                min_value=5,
                max_value=50,
                value=10
            )

            # Cálculo del TOP N
            top_reviews = (
                df.groupby('neighbourhood_cleansed')['number_of_reviews']
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            # Gráfica de barras descendente
            fig_bar_reviews = px.bar(
                top_reviews.sort_values("number_of_reviews", ascending=True),  # Reordenar para gráfica horizontal
                x='number_of_reviews',
                y='neighbourhood_cleansed',
                orientation='h',
                title=f"TOP {top_n} Barrios con Mayor Demanda (Por Reviews)"
            )

            st.plotly_chart(fig_bar_reviews, use_container_width=True)

            

        # --- PREGUNTA 4: Precio vs Amenities ---
        st.subheader("4. Precio justo respecto a propiedades similares del barrio")
        
        with col1:
            fig_violin = px.violin(
            df_filtered,
            x="host_is_superhost",
            y="price",
            box=True,
            color="host_is_superhost",
            color_discrete_map={0: "#FFB6C1", 1: "#87CEFA"},
            labels={
                "host_is_superhost": "Es Superhost (0=No, 1=Sí)",
                "price": "Precio (€)"
            },
            title="Impacto de ser Superhost en el Precio"
        )
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
    
        # --- PREGUNTA 5: Turismo Alto vs Residencial ---
        st.subheader("5. Variación de Precio: Barrios Turísticos vs. Residenciales")
        # Agrupar por grupo de vecindario (Distrito)
        price_by_group = df.groupby('neighbourhood_cleansed')['price'].mean().sort_values().reset_index()
        
        fig_group_price = px.bar(
            price_by_group,
            x='neighbourhood_cleansed',
            y='price',
            color='price',
            title="Precio Promedio por Distrito",
            labels={"neighbourhood_cleansed": "Distrito", "price": "Precio Promedio (€)"}
        )
        st.plotly_chart(fig_group_price, use_container_width=True)

    # =============================================================================
    # PESTAÑA 2: MODELO PREDICTIVO (Regresión)
    # =============================================================================
    with tab2:
        st.header("Predicción de Precio Sugerido")
        st.markdown("Utiliza este modelo para estimar el precio ideal de una propiedad basado en sus características.")

        st.markdown("Selecciona el modelo que deseas usar:")
        # Inputs del usuario para la predicción
        pred_tab1, pred_tab2 = st.tabs(["TEC", "Uniandes"])
        # ==========================================================
    # ---------------------- MODELO TEC -------------------------
    # ==========================================================
    with pred_tab1:
        st.subheader("Modelo Predictivo - TEC")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            input_accommodates = st.number_input(
                "Número de Personas (Accommodates)", 
                min_value=1, 
                max_value=16, 
                value=2,
                key="tec_accom"
            )
            input_bathrooms = st.number_input(
                "Baños", 
                min_value=0.0, 
                max_value=10.0, 
                value=1.0,
                key="tec_baths"
            )
            input_room_type = st.selectbox(
                "Tipo de Habitación", 
                df['room_type'].unique(),
                key='tec_room'
            )

        with col_p2:
            input_bedrooms = st.number_input(
                "Habitaciones", 
                min_value=0, 
                max_value=10, 
                value=1,
                key="tec_beds"
            )
            input_neigh = st.selectbox(
                "Barrio", 
                df['neighbourhood_cleansed'].unique(), 
                key='tec_neigh'
            )

        # ------- Botón de predicción --------
        if st.button("Calcular Precio Sugerido - TEC"):
            if reg_model:
                # Aquí deberías construir el array igual que en el notebook
                prediction_mock = (input_accommodates * 22) + (input_bedrooms * 18) + 25
                st.success(f"Precio sugerido (TEC): **€ {prediction_mock:.2f}** por noche.")
            else:
                st.warning("Modelo TEC no encontrado.")

        # ------- Métricas del modelo --------
        st.subheader("Desempeño del Modelo TEC")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R2 Score", "0.75 (Ejemplo)")
        col_m2.metric("RMSE", "€ 15.40 (Ejemplo)")
        

    # ==========================================================
    # -------------------- MODELO UNIANDES ----------------------
    # ==========================================================
    with pred_tab2:
        st.subheader("Modelo Predictivo - Uniandes")

        col_u1, col_u2 = st.columns(2)

        with col_u1:
            input_accommodates_u = st.number_input(
                "Número de Personas (Accommodates)", 
                min_value=1, 
                max_value=16, 
                value=2,
                key="uni_accom"
            )
            input_bathrooms_u = st.number_input(
                "Baños", 
                min_value=0.0, 
                max_value=10.0, 
                value=1.0,
                key="uni_baths"
            )
            input_room_type_u = st.selectbox(
                "Tipo de Habitación", 
                df['room_type'].unique(),
                key='uni_room'
            )

        with col_u2:
            input_bedrooms_u = st.number_input(
                "Habitaciones", 
                min_value=0, 
                max_value=10, 
                value=1,
                key="uni_beds"
            )
            input_neigh_u = st.selectbox(
                "Barrio", 
                df['neighbourhood_cleansed'].unique(), 
                key='uni_neigh'
            )

        if st.button("Calcular Precio Sugerido - Uniandes"):
            prediction_mock = (input_accommodates_u * 18) + (input_bedrooms_u * 20) + 40
            st.success(f"Precio sugerido (Uniandes): **€ {prediction_mock:.2f}** por noche.")

        st.subheader("Desempeño del Modelo Uniandes")
        col_m3, col_m4 = st.columns(2)
        col_m3.metric("R2 Score", "0.78 (Ejemplo)")
        col_m4.metric("RMSE", "€ 13.90 (Ejemplo)")

    # =============================================================================
    # PESTAÑA 3: MODELO DE CLASIFICACIÓN
    # =============================================================================
    with tab3:
        st.header("Clasificación de Riesgo / Tipo")
        st.markdown("Selecciona el modelo de clasificación que deseas usar:")

        clf_tab1, clf_tab2 = st.tabs(["TEC", "Uniandes"])

        # ==========================================================
        # -------------------- CLASIFICADOR TEC ---------------------
        # ==========================================================
        with clf_tab1:
            st.subheader("Clasificador - TEC")

            input_reviews_TEC = st.slider(
                "Número de Reseñas", 
                0, 500, 20,
                key="clf_tec_reviews"
            )
            input_score_TEC = st.slider(
                "Puntuación (0-5)", 
                0.0, 5.0, 4.5,
                key="clf_tec_score"
            )

            if st.button("Clasificar (TEC)"):
                prob = np.random.rand()
                clase = "Alta Rentabilidad" if prob > 0.5 else "Baja Rentabilidad"

                st.success(f"Predicción TEC: **{clase}** ({prob:.0%})")

            st.subheader("Matriz de Confusión - TEC")
            st.info("Aquí se mostrará la matriz del modelo TEC.")

        # ==========================================================
        # ---------------- CLASIFICADOR UNIANDES -------------------
        # ==========================================================
        with clf_tab2:
            st.subheader("Clasificador - Uniandes")

            input_reviews_Uni = st.slider(
                "Número de Reseñas", 
                0, 500, 20,
                key="clf_uni_reviews"
            )
            input_score_Uni = st.slider(
                "Puntuación (0-5)", 
                0.0, 5.0, 4.0,
                key="clf_uni_score"
            )

            if st.button("Clasificar (Uniandes)"):
                prob = np.random.rand()
                clase = "Alta Rentabilidad" if prob > 0.55 else "Baja Rentabilidad"
                st.success(f"Predicción Uniandes: **{clase}** ({prob:.0%})")

            st.subheader("Matriz de Confusión - Uniandes")
            st.info("Aquí se mostrará la matriz del modelo Uniandes.")
