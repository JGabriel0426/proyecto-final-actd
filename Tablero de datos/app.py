import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import ast
from tensorflow import keras
import joblib
import tensorflow as tf
from tensorflow import keras

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard Airbnb Barcelona",
    layout="wide"
)
def load_classification_model():
    try:
        model = keras.models.load_model("modelo_clasificacion.keras")
        scaler = joblib.load("scaler_clasificacion.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None, None


features = ['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'maximum_nights', 'instant_bookable', 'calculated_host_listings_count', 'n_amenities', 'n_verifications', 'host_years', 'neighbourhood_cleansed_freq', 'room_type_Hotel room', 'room_type_Private room', 'room_type_Shared room', 'neighbourhood_group_cleansed_Eixample', 'neighbourhood_group_cleansed_Gràcia', 'neighbourhood_group_cleansed_Horta-Guinardó', 'neighbourhood_group_cleansed_Les Corts', 'neighbourhood_group_cleansed_Nou Barris', 'neighbourhood_group_cleansed_Sant Andreu', 'neighbourhood_group_cleansed_Sant Martí', 'neighbourhood_group_cleansed_Sants-Montjuïc', 'neighbourhood_group_cleansed_Sarrià-Sant Gervasi', 'property_type_clean_Entire rental unit', 'property_type_clean_Entire serviced apartment', 'property_type_clean_Other', 'property_type_clean_Private room in rental unit', 'property_type_clean_Room in hotel']
fetures_reg_gabo=[
'host_response_time', 'host_response_rate', 'host_acceptance_rate',
'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
'beds', 'minimum_nights', 'maximum_nights', 'availability_30',
'availability_365', 'number_of_reviews', 'availability_eoy',
'estimated_occupancy_l365d', 'review_scores_rating',
'review_scores_accuracy', 'review_scores_cleanliness',
'review_scores_checkin', 'review_scores_communication',
'review_scores_location', 'review_scores_value', 'instant_bookable',
'calculated_host_listings_count', 'reviews_per_month', 'n_amenities',
'n_verifications', 'host_years', 'neighbourhood_cleansed_freq',
'room_type_Hotel room', 'room_type_Private room',
'room_type_Shared room', 'neighbourhood_group_cleansed_Eixample',
'neighbourhood_group_cleansed_Gràcia',
'neighbourhood_group_cleansed_Horta-Guinardó',
'neighbourhood_group_cleansed_Les Corts',
'neighbourhood_group_cleansed_Nou Barris',
'neighbourhood_group_cleansed_Sant Andreu',
'neighbourhood_group_cleansed_Sant Martí',
'neighbourhood_group_cleansed_Sants-Montjuïc',
'neighbourhood_group_cleansed_Sarrià-Sant Gervasi'
]
mean_values = {
    'host_response_time': 0.4093179276928811,
    'host_response_rate': 93.302274,
    'host_acceptance_rate': 85.256504,
    'host_is_superhost': 0.241073,
    'host_has_profile_pic': 0.962579,
    'host_identity_verified': 0.954379,
    'latitude': 41.392294,
    'longitude': 2.166916,
    'minimum_nights': 14.928513,
    'maximum_nights': 491.240030,
    'instant_bookable': 0.42996645546030565,
    'calculated_host_listings_count': 60.1769660827432,
    'n_verifications': 2.096011926947447,
    'host_years': 8.158900841932205,
    'neighbourhood_cleansed_freq': 0.05039359579453739,

    # Valores agregados 
    'accommodates': 3.441893,
    'bathrooms': 1.418785,
    'bedrooms': 1.759076,
    'beds': 2.392546,
    'price': None,   
    'availability_30': 9.076631,
    'availability_365': 225.156541,
    'number_of_reviews': 64.615729,
    'availability_eoy': 58.958330,
    'estimated_occupancy_l365d': 98.418412,
    'estimated_revenue_l365d': 28,  
    'review_scores_rating': 4.627713,
    'review_scores_accuracy': 4.675666,
    'review_scores_cleanliness': 4.650232,
    'review_scores_checkin': 4.745778,
    'review_scores_communication': 4.746362,
    'review_scores_location': 4.768934,
    'review_scores_value': 4.475582,
    'reviews_per_month': 1.346702,
    'n_amenities': 28.576966,

    # Dummies de tipo de habitación
    'room_type_Hotel room': 0.003354,
    'room_type_Private room': 0.304137,
    'room_type_Shared room': 0.005367,

    # Dummies de barrio
    'neighbourhood_group_cleansed_Eixample': 0.351174,
    'neighbourhood_group_cleansed_Gràcia': 0.089676,
    'neighbourhood_group_cleansed_Horta-Guinardó': 0.027581,
    'neighbourhood_group_cleansed_Les Corts': 0.021543,
    'neighbourhood_group_cleansed_Nou Barris': 0.010734,
    'neighbourhood_group_cleansed_Sant Andreu': 0.015878,
    'neighbourhood_group_cleansed_Sant Martí': 0.094223,
    'neighbourhood_group_cleansed_Sants-Montjuïc': 0.100485,
    'neighbourhood_group_cleansed_Sarrià-Sant Gervasi': 0.061573
}
b0 = 10457.954365130736
betas = np.array([
    -9.02810491e+00, -3.12327409e-01,  1.46533482e-01,  1.17525558e+01,
    2.08003440e+01, -1.70399641e+01, -2.53029697e+02,  2.60723584e+01,
    1.17081171e+01,  5.00190209e+00, -1.73224601e+00, -1.03611845e+00,
    -1.36345353e+00,  4.55336990e-03,  8.05642063e-01, -1.29135134e-02,
    -3.45600665e-02,  2.89377904e-02, -4.44931407e-01,  3.09771903e-03,
    6.83398356e+00,  5.49715602e-01,  9.10411553e+00, -1.83099943e-01,
    -5.00857484e+00,  2.45370661e+00, -1.55987707e+00,  1.03861585e+01,
    2.49475471e-02, -1.02880479e+00,  2.12859479e-01,  5.20535150e+00,
    -9.83317282e-02,  1.57452211e+02,  1.71970348e+01, -2.55388919e+01,
    -8.57023275e+01,  1.05022965e+01,  6.68717834e+00,  7.86499533e+00,
    6.61769528e+00,  7.22890110e+00,  1.31124307e+01,  1.88588806e+01,
    -5.44108827e+00,  2.14158907e+01
], dtype=float)
features_tec_clsf = [
    'host_response_time', 'host_response_rate', 'host_acceptance_rate',
    'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified',
    'latitude', 'longitude', 'accommodates', 'bathrooms', 'bedrooms',
    'beds', 'minimum_nights', 'maximum_nights', 'availability_30',
    'availability_365', 'number_of_reviews', 'availability_eoy',
    'estimated_occupancy_l365d', 'estimated_revenue_l365d',
    'review_scores_rating', 'review_scores_accuracy',
    'review_scores_cleanliness', 'review_scores_checkin',
    'review_scores_communication', 'review_scores_location',
    'review_scores_value', 'instant_bookable',
    'calculated_host_listings_count', 'reviews_per_month', 'n_amenities',
    'n_verifications', 'host_years', 'neighbourhood_cleansed_freq',
    'room_type_Hotel room', 'room_type_Private room',
    'room_type_Shared room', 'neighbourhood_group_cleansed_Eixample',
    'neighbourhood_group_cleansed_Gràcia',
    'neighbourhood_group_cleansed_Horta-Guinardó',
    'neighbourhood_group_cleansed_Les Corts',
    'neighbourhood_group_cleansed_Nou Barris',
    'neighbourhood_group_cleansed_Sant Andreu',
    'neighbourhood_group_cleansed_Sant Martí',
    'neighbourhood_group_cleansed_Sants-Montjuïc',
    'neighbourhood_group_cleansed_Sarrià-Sant Gervasi'
]
def build_input_vector_tec(user_inputs: dict):
 
    row = dict.fromkeys(features_tec_clsf, 0.0) 
    for k, v in user_inputs.items():
        if k in row:
            row[k] = float(v) if v is not None else 0.0

    # one-hot room_type
    rt = user_inputs.get("room_type", None)
    if rt is not None:
        col_rt = f"room_type_{rt}"
        if col_rt in row:
            row[col_rt] = 1.0

    # one-hot neighbourhood
    ng = user_inputs.get("neigh_group", None)
    if ng is not None:
        col_ng = f"neighbourhood_group_cleansed_{ng}"
        if col_ng in row:
            row[col_ng] = 1.0

    
    X = np.array([row[feat] for feat in features_tec_clsf], dtype=float)
    return X

def predict_price_tec(user_inputs: dict):
    X = build_input_vector_tec(user_inputs)      # vector 46
    precio = b0 + float(np.dot(betas, X))
    return precio
etas=[ 8.58244554e-04,  1.05651822e-02, -7.67630261e-03,
         8.40717038e-04,  1.79300661e-04,  3.01006363e-04,
         9.41878191e-03,  4.93060538e-04, -3.31310336e-04,
         6.51420637e-04,  4.39988375e-04, -3.44676285e-04,
        -1.89424311e-04,  2.43030705e-02, -6.46263915e-04,
         6.82203335e-04,  5.57385785e-05,  4.00410496e-03,
        -5.58431220e-03, -6.04704581e-03,  2.08418866e-05,
         2.84825661e-03,  2.70932123e-03,  2.63007279e-03,
         2.38245471e-03,  2.48074335e-03,  2.00193025e-03,
        -4.92093295e-04, -1.23382410e-03, -5.21457281e-04,
         8.25071339e-03,  3.49937001e-04,  5.93077039e-04,
         1.69108738e-05, -3.82437948e-06,  4.64624274e-04,
        -6.88989821e-06,  6.51355008e-05,  1.19374601e-05,
         4.62818946e-06,  1.95486029e-05,  2.56150719e-05,
         2.65761553e-05,  4.61474052e-07, -4.42354893e-05,
         5.79308790e-05]
def _sigmoid(z: float) -> float:
    """Sigmoide numericamente estable."""
    # protección contra overflow:
    if z >= 0:
        ez = np.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = np.exp(z)
        return ez / (1.0 + ez)
def predict_clsf_tec(user_inputs: dict):
    X = build_input_vector_tec(user_inputs) 
    z = 0 + float(np.dot(etas, X))     # vector 46
    p = _sigmoid(z)
    
    cls = 1 if p >= 0.5 else 0 
    return p,cls
def build_input_vector(user_inputs: dict):
    row = dict.fromkeys(features, 0)

    # Llenar valores numéricos
    for key, value in user_inputs.items():
        if key in row:
            row[key] = value

    # One-hot de room_type
    if f"room_type_{user_inputs['room_type']}" in row:
        row[f"room_type_{user_inputs['room_type']}"] = 1

    # One-hot de grupo barrio
    if f"neighbourhood_group_cleansed_{user_inputs['neigh_group']}" in row:
        row[f"neighbourhood_group_cleansed_{user_inputs['neigh_group']}"] = 1

    # One-hot de tipo de propiedad
    if f"property_type_clean_{user_inputs['property_type_clean']}" in row:
        row[f"property_type_clean_{user_inputs['property_type_clean']}"] = 1

    return pd.DataFrame([row], columns=features)
def build_input_vector_type2(user_inputs: dict):
    row = dict.fromkeys(fetures_reg_gabo, 0)

    # Llenar valores numéricos
    for key, value in user_inputs.items():
        if key in row:
            row[key] = value

    # One-hot de room_type
    if f"room_type_{user_inputs['room_type']}" in row:
        row[f"room_type_{user_inputs['room_type']}"] = 1

    # One-hot de grupo barrio
    if f"neighbourhood_group_cleansed_{user_inputs['neigh_group']}" in row:
        row[f"neighbourhood_group_cleansed_{user_inputs['neigh_group']}"] = 1

    # One-hot de tipo de propiedad
    if f"property_type_clean_{user_inputs['property_type_clean']}" in row:
        row[f"property_type_clean_{user_inputs['property_type_clean']}"] = 1

    return pd.DataFrame([row], columns=fetures_reg_gabo)
model_clf, scaler_clf = load_classification_model()
def predecir_clasificacion(user_inputs):
    if model_clf is None or scaler_clf is None:
        return "Error", 0.0
    
    X = build_input_vector(user_inputs)
    # print(X)
    # 'ayuda'
    # X

    try:
        X_scaled = scaler_clf.transform(X)
    except Exception as e:
        st.error(f"Error escalando las entradas: {e}")
        return "Error", 0.0

    try:
        prob = float(model_clf.predict(X_scaled)[0][0])
    except Exception as e:
        st.error(f"Error haciendo la predicción: {e}")
        return "Error", 0.0

    clase = "Alta Rentabilidad" if prob >= 0.5 else "Baja Rentabilidad"
    return clase, prob

def load_price_model():
    try:
        model = keras.models.load_model("modelo_regresion.keras")
        scaler = joblib.load("scaler_regresion.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error cargando el modelo de precio: {e}")
        return None, None

model_price, scaler_price = load_price_model()
def predecir_precio(user_inputs):
    if model_price is None or scaler_price is None:
        return 0.0
    # user_inputs
    # Crear vector de inputs igual que para clasificación
    X = build_input_vector_type2(user_inputs)
    # X
    try:
        X_scaled = scaler_price.transform(X)
    except Exception as e:
        st.error(f"Error escalando las entradas: {e}")
        return 0.0

    try:
        precio_pred = np.expm1(float(model_price.predict(X_scaled)[0][0].flatten()))
    except Exception as e:
        st.error(f"Error haciendo la predicción de precio: {e}")
        return 0.0

    return precio_pred
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
        st.subheader("Modelo Predictivo - TEC (Regresión Lineal)")
        st.write("Introduce las características del inmueble (TEC).")

        c1, c2 = st.columns(2)
        with c1:
            tec_accommodates = st.number_input("Acomoda (TEC):", 1, 16, 2, key="tec_accommodates")
            tec_bathrooms = st.number_input("Baños (TEC):", 0.0, 10.0, 1.0, key="tec_bathrooms")
            tec_bedrooms = st.number_input("Habitaciones (TEC):", 0, 10, 1, key="tec_bedrooms")
            tec_beds = st.number_input("Camas (TEC):", 1, 10, 1, key="tec_beds")
            tec_price_input = st.number_input("Precio actual / referencia (TEC):", 0.0, 10000.0, 50.0, key="tec_price_input")  # si el modelo usa price como predictor
            n_amenities = st.number_input("Número de amenities:", 0, 50, 10, key="clf_n_amenities_tec")
        with c2:
            tec_room_type = st.selectbox("Tipo habitación (TEC):", ["Hotel room", "Private room", "Shared room"], key="tec_room_type")
            tec_neigh_group = st.selectbox("Distrito (TEC):", [
                "Eixample", "Gràcia", "Horta-Guinardó", "Les Corts",
                "Nou Barris", "Sant Andreu", "Sant Martí",
                "Sants-Montjuïc", "Sarrià-Sant Gervasi"
            ], key="tec_neigh_group")

        # filas siguientes (puedes ajustar valores por defecto según mean_values)
        c3, c4, c5 = st.columns(3)
        with c3:
            tec_host_response_time = st.number_input("Host response time (TEC):", 0.0, 100.0, float(mean_values.get("host_response_time", 0.0)), key="tec_host_response_time")
            tec_host_response_rate = st.number_input("Host response rate (TEC):", 0.0, 100.0, float(mean_values.get("host_response_rate", 0.0)), key="tec_host_response_rate")
            tec_host_acceptance_rate = st.number_input("Host acceptance rate (TEC):", 0.0, 100.0, float(mean_values.get("host_acceptance_rate", 0.0)), key="tec_host_acceptance_rate")
        with c4:
            tec_host_is_superhost = st.selectbox("Es superhost (TEC):", [0,1], index=int(round(mean_values.get("host_is_superhost",0))), key="tec_host_is_superhost")
            tec_host_has_profile_pic = st.selectbox("Host tiene foto (TEC):", [0,1], index=int(round(mean_values.get("host_has_profile_pic",1))), key="tec_host_has_profile_pic")
            tec_host_identity_verified = st.selectbox("Host identidad verificada (TEC):", [0,1], index=int(round(mean_values.get("host_identity_verified",1))), key="tec_host_identity_verified")
        with c5:
            tec_latitude = st.number_input("Latitud (TEC):", -90.0, 90.0, float(mean_values.get("latitude", 0.0)), key="tec_latitude")
            tec_longitude = st.number_input("Longitud (TEC):", -180.0, 180.0, float(mean_values.get("longitude", 0.0)), key="tec_longitude")
            tec_accommodates_2 = st.number_input("Acomoda (dup) (TEC):", 1, 16, 2, key="tec_accommodates_2")  # si quieres duplicar inputs para UX, no obligatorio
        c6, c7, c8 = st.columns(3)

        with c6:
            tec_minimum_nights = st.number_input(
                "Estancia mínima (TEC):",
                1, 500,
                int(mean_values.get("minimum_nights", 1)),
                key="tec_minimum_nights"
            )
            tec_maximum_nights = st.number_input(
                "Estancia máxima (TEC):",
                1, 500,
                int(mean_values.get("maximum_nights", 30)),
                key="tec_maximum_nights"
            )
            tec_availability_30 = st.number_input(
                "Disponibilidad 30 días (TEC):",
                0, 30,
                int(mean_values.get("availability_30", 0)),
                key="tec_availability_30"
            )

        with c7:
            tec_availability_365 = st.number_input(
                "Disponibilidad 365 días (TEC):",
                0, 365,
                int(mean_values.get("availability_365", 0)),
                key="tec_availability_365"
            )
            tec_number_reviews = st.number_input(
                "Número de reviews (TEC):",
                0, 1000,
                int(mean_values.get("number_of_reviews", 0)),
                key="tec_number_reviews"
            )
            tec_availability_eoy = st.number_input(
                "Availability end of year (TEC):",
                0, 365,
                int(mean_values.get("availability_eoy", 0)),
                key="tec_availability_eoy"
            )

        with c8:
            tec_estimated_occupancy = st.number_input(
                "Ocupación estimada 365d (TEC):",
                0.0, 100.0,
                float(mean_values.get("estimated_occupancy_l365d", 0.0)),
                key="tec_estimated_occupancy"
            )
            tec_estimated_revenue = st.number_input(
                "Ingresos estimados 365d (TEC):",
                0.0, 500000.0,
                float(mean_values.get("estimated_revenue_l365d", 0.0)),
                key="tec_estimated_revenue"
            )
            tec_reviews_per_month = st.number_input(
                "Reviews por mes (TEC):",
                0.0, 50.0,
                float(mean_values.get("reviews_per_month", 0.0)),
                key="tec_reviews_per_month"
            )

        # -------- Reviews scores --------
        c9, c10, c11 = st.columns(3)

        with c9:
            tec_score_rating = st.number_input("Rating (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_rating", 4.0)),
                key="tec_score_rating")
            tec_score_accuracy = st.number_input("Accuracy (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_accuracy", 4.0)),
                key="tec_score_accuracy")
            tec_score_cleanliness = st.number_input("Cleanliness (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_cleanliness", 4.0)),
                key="tec_score_cleanliness")
            tec_neighbourhood_freq = st.number_input(
                "Frecuencia relativa del barrio (TEC):",
                0.0, 100.0,
                float(mean_values.get("neighbourhood_cleansed_freq", 0.0)),
                key="tec_neighbourhood_freq")
        with c10:
            tec_score_checkin = st.number_input("Check-in (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_checkin", 4.0)),
                key="tec_score_checkin")
            tec_score_comm = st.number_input("Communication (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_communication", 4.0)),
                key="tec_score_comm")
            tec_score_location = st.number_input("Location (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_location", 4.0)),
                key="tec_score_location")

        with c11:
            tec_score_value = st.number_input("Value (TEC):", 0.0, 5.0,
                float(mean_values.get("review_scores_value", 4.0)),
                key="tec_score_value")
            tec_host_years = st.number_input("Años del host (TEC):",
                0, 50,
                int(mean_values.get("host_years", 1)),
                key="tec_host_years")
            tec_n_verifications = st.number_input("N° verificaciones host (TEC):",
                0, 20,
                int(mean_values.get("n_verifications", 0)),
                key="tec_n_verifications")
            tec_calculated_host_listings_count = st.number_input(
            "N° propiedades del host (TEC):",
            0, 1000,
            int(mean_values.get("calculated_host_listings_count", 1)),
            key="tec_calculated_host_listings_count")

            
        # Puedes añadir más inputs según necesites (availability, reviews, scores...)
        # Para brevedad, usaremos valores por defecto extraídos de mean_values si no se piden explícitamente
        if st.button("Predecir Precio (TEC)"):
            # Construir user_inputs con las 46 variables (usar mean_values cuando falten)
            ui = {
                "host_response_time": tec_host_response_time,
                "host_response_rate": tec_host_response_rate,
                "host_acceptance_rate": tec_host_acceptance_rate,
                "host_is_superhost": tec_host_is_superhost,
                "host_has_profile_pic": tec_host_has_profile_pic,
                "host_identity_verified": tec_host_identity_verified,
                "latitude": tec_latitude,
                "longitude": tec_longitude,
                "accommodates": tec_accommodates,
                "bathrooms": tec_bathrooms,
                "bedrooms": tec_bedrooms,
                "beds": tec_beds,
                "price": tec_price_input,
                "minimum_nights": tec_minimum_nights,
                "maximum_nights": tec_maximum_nights,
                "availability_30": tec_availability_30,
                "availability_365": tec_availability_365,
                "number_of_reviews": tec_number_reviews,
                "availability_eoy": tec_availability_eoy,
                "estimated_occupancy_l365d": tec_estimated_occupancy,
                "estimated_revenue_l365d": tec_estimated_revenue,
                "review_scores_rating": tec_score_rating,
                "review_scores_accuracy": tec_score_accuracy,
                "review_scores_cleanliness": tec_score_cleanliness,
                "review_scores_checkin": tec_score_checkin,
                "review_scores_communication": tec_score_comm,
                "review_scores_location": tec_score_location,
                "review_scores_value": tec_score_value,
                "instant_bookable": 1,
                "calculated_host_listings_count": tec_calculated_host_listings_count,
                "reviews_per_month": tec_reviews_per_month,
                "n_amenities": n_amenities,
                "n_verifications": tec_n_verifications,
                "host_years": tec_host_years,
                "neighbourhood_cleansed_freq": tec_neighbourhood_freq,
                "room_type": tec_room_type,
                "neigh_group": tec_neigh_group
            }

            precio_tec = predict_price_tec(ui)
            st.success(f"Precio estimado (TEC): **€ {precio_tec:.2f}**")
        

    # ==========================================================
    # -------------------- MODELO UNIANDES ----------------------
    # ==========================================================
    with pred_tab2:
        st.subheader("Modelo Predictivo - Uniandes")

        st.write("Introduce las características del inmueble:")

        col1, col2 = st.columns(2)

        with col1:
            accommodates = st.number_input("Acomoda:", 1, 16, 2, key="clf_accommodates_1")
            bathrooms = st.number_input("Baños:", 0.0, 10.0, 1.0, key="clf_bathrooms_1")
            bedrooms = st.number_input("Habitaciones:", 0, 10, 1, key="clf_bedrooms_1")
            beds = st.number_input("Camas:", 1, 10, 1, key="clf_beds_1")
            n_amenities = st.number_input("Número de amenities:", 0, 50, 10, key="clf_n_amenities_1")

        with col2:
            room_type = st.selectbox(
                "Tipo habitación:",
                ["Hotel room", "Private room", "Shared room"],
                key="clf_room_type_1"
            )
            neigh_group = st.selectbox(
                "Distrito:",
                [
                    "Eixample", "Gràcia", "Horta-Guinardó", "Les Corts",
                    "Nou Barris", "Sant Andreu", "Sant Martí",
                    "Sants-Montjuïc", "Sarrià-Sant Gervasi"
                ],
                key="clf_neigh_group_1"
            )
            property_type_clean = st.selectbox("Tipo de propiedad:", [
                    "Entire rental unit", "Entire serviced apartment", "Other",
                    "Private room in rental unit", "Room in hotel"
                ],key="property_type_clean_1")

        # Parámetros del host
        st.markdown("---")
        st.markdown("### Parámetros del Host (Valores sugeridos del dataset)")
        colh1, colh2 = st.columns(2)

        with colh1:
            host_response_time = st.number_input("Tiempo de respuesta del host (horas)", 0.0, 72.0, float(mean_values["host_response_time"]), key="host_response_time_1")
            host_response_rate = st.number_input("Tasa de respuesta del host (0-100)", 0.0, 100.0, float(mean_values["host_response_rate"]), key="host_response_rate_1")
            host_acceptance_rate = st.number_input("Tasa de aceptación (0-100)", 0.0, 100.0, float(mean_values["host_acceptance_rate"]), key="host_acceptance_rate_1")
            host_is_superhost = st.selectbox("¿Es Superhost?", [0, 1], index=int(round(mean_values["host_is_superhost"])), key="host_is_superhost_1")
            host_has_profile_pic = st.selectbox("¿Tiene foto de perfil?", [0, 1], index=int(round(mean_values["host_has_profile_pic"])), key="host_has_profile_pic_1")

        with colh2:
            host_identity_verified = st.selectbox("¿Identidad verificada?", [0, 1], index=int(round(mean_values["host_identity_verified"])), key="host_identity_verified_1")
            minimum_nights = st.number_input("Mínimo de noches", 1, 365, int(mean_values["minimum_nights"]), key="minimum_nights_1")
            maximum_nights = st.number_input("Máximo de noches", 1, 9999, int(mean_values["maximum_nights"]), key="maximum_nights_1")
            calculated_host_listings_count = st.number_input("Número de propiedades del host", 0, 9999, int(mean_values["calculated_host_listings_count"]), key="calculated_host_listings_count_1")
            n_verifications = st.number_input("Número de verificaciones", 0, 10, int(mean_values["n_verifications"]), key="n_verifications_1")

        st.markdown("### Ubicación, Frecuencia del Barrio y otros Parámetros")
        coll1, coll2, coll3 = st.columns(3)

        # ------- FILA 1 -------
        with coll1:
            latitude = st.number_input("Latitud", -90.0, 90.0,
                                    float(mean_values["latitude"]), key="latitude_input")

            host_years = st.number_input("Años del host en plataforma", 0, 20,
                                        int(mean_values["host_years"]), key="host_years_input")

            availability_30 = st.number_input("Disponibilidad 30 días", 0, 30,
                                            int(mean_values["availability_30"]), key="availability_30_input")

            review_scores_cleanliness = st.number_input("Score limpieza", 0.0, 5.0,
                                                        float(mean_values["review_scores_cleanliness"]), key="review_scores_cleanliness_input")

        with coll2:
            longitude = st.number_input("Longitud", -180.0, 180.0,
                                        float(mean_values["longitude"]), key="longitude_input")

            neighbourhood_cleansed_freq = st.number_input("Frecuencia relativa del barrio", 0.0, 0.5,
                                                        float(mean_values["neighbourhood_cleansed_freq"]), key="neighbourhood_cleansed_freq_input")

            availability_365 = st.number_input("Disponibilidad 365 días", 0, 400,
                                            int(mean_values["availability_365"]), key="availability_365_input")

            review_scores_checkin = st.number_input("Score check-in", 0.0, 10.0,
                                                    float(mean_values["review_scores_checkin"]), key="review_scores_checkin_input")

        with coll3:
            number_of_reviews = st.number_input("Número de reviews", 0, 500,
                                                int(mean_values["number_of_reviews"]), key="number_of_reviews_input")

            availability_eoy = st.number_input("Disponibilidad fin de año", 0, 300,
                                            int(mean_values["availability_eoy"]), key="availability_eoy_input")

            estimated_occupancy_l365d = st.number_input("Ocupación estimada 365 días (%)", 0.0, 100.0,
                                                        float(mean_values["estimated_occupancy_l365d"]), key="estimated_occupancy_input")

            review_scores_communication = st.number_input("Score comunicación", 0.0, 5.0,
                                                        float(mean_values["review_scores_communication"]), key="review_scores_communication_input")


        # ------- FILA 2 --------
        coll4, coll5, coll6 = st.columns(3)

        with coll4:
            reviews_per_month = st.number_input("Reviews por mes", 0.0, 50.0,
                                                float(mean_values["reviews_per_month"]), key="reviews_per_month_input")

            review_scores_location = st.number_input("Score ubicación", 0.0, 5.0,
                                                    float(mean_values["review_scores_location"]), key="review_scores_location_input")

        with coll5:
            estimated_revenue_l365d = st.number_input("Ingresos estimados 365 días", 0.0, 200000.0,
                                                    float(mean_values["estimated_revenue_l365d"]), key="estimated_revenue_input")

            review_scores_accuracy = st.number_input("Score accuracy", 0.0, 5.0,
                                                    float(mean_values["review_scores_accuracy"]), key="review_scores_accuracy_input")

        with coll6:
            review_scores_rating = st.number_input("Score rating", 0.0, 5.0,
                                                float(mean_values["review_scores_rating"]), key="review_scores_rating_input")

            review_scores_value = st.number_input("Score value", 0.0, 5.0,
                                                float(mean_values["review_scores_value"]), key="review_scores_value_input")

        # Botón y predicción
        if st.button("Predecir Precio"):
            user_inputs = {
                "accommodates": accommodates,
                "bathrooms": bathrooms,
                "bedrooms": bedrooms,
                "beds": beds,
                "n_amenities": n_amenities,
                "room_type": room_type,
                "neigh_group": neigh_group,
                "property_type_clean": property_type_clean,
            }
            user_inputs.update({                
                "host_response_time": host_response_time,
                "host_response_rate": host_response_rate,
                "host_acceptance_rate": host_acceptance_rate,
                "host_is_superhost": host_is_superhost,
                "host_has_profile_pic": host_has_profile_pic,
                "host_identity_verified": host_identity_verified,
                "latitude": latitude,
                "longitude": longitude,
                "minimum_nights": minimum_nights,
                "maximum_nights": maximum_nights,
                "instant_bookable": 1,
                "calculated_host_listings_count": calculated_host_listings_count,
                "n_verifications": n_verifications,
                "host_years": host_years,
                "neighbourhood_cleansed_freq": neighbourhood_cleansed_freq,
                "reviews_per_month": reviews_per_month,
                "availability_30": availability_30,
                "availability_365": availability_365,
                "number_of_reviews": number_of_reviews,
                "availability_eoy": availability_eoy,
                "estimated_occupancy_l365d": estimated_occupancy_l365d,
                # "estimated_revenue_l365d": estimated_revenue_l365d,
                "review_scores_rating": review_scores_rating,
                "review_scores_accuracy": review_scores_accuracy,
                "review_scores_cleanliness": review_scores_cleanliness,
                "review_scores_checkin": review_scores_checkin,
                "review_scores_communication": review_scores_communication,
                "review_scores_location": review_scores_location,
                "review_scores_value": review_scores_value
            })

            precio = predecir_precio(user_inputs)
            st.success(f"Precio estimado de tu inmueble: **{precio:.2f} €**")


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
            st.subheader("Modelo Predictivo - TEC (Clasificación Logística)")
            st.write("Introduce las características del inmueble para clasificar su nivel TEC.")

            c1, c2 = st.columns(2)
            with c1:
                clf_accommodates = st.number_input("Acomoda (TEC):", 1, 16, 2, key="clf_accommodates")
                clf_bathrooms = st.number_input("Baños (TEC):", 0.0, 10.0, 1.0, key="clf_bathrooms")
                clf_bedrooms = st.number_input("Habitaciones (TEC):", 0, 10, 1, key="clf_bedrooms")
                clf_beds = st.number_input("Camas (TEC):", 1, 10, 1, key="clf_beds")
                clf_price = st.number_input("Precio referencia (TEC):", 0.0, 10000.0, 50.0, key="clf_price")
                clf_n_amenities = st.number_input("Número de amenities:", 0, 50, 10, key="clf_n_amenities")
            with c2:
                clf_room_type = st.selectbox("Tipo habitación (TEC):",
                    ["Hotel room", "Private room", "Shared room"],
                    key="clf_room_type")
                clf_neigh_group = st.selectbox("Distrito (TEC):", [
                    "Eixample", "Gràcia", "Horta-Guinardó", "Les Corts",
                    "Nou Barris", "Sant Andreu", "Sant Martí",
                    "Sants-Montjuïc", "Sarrià-Sant Gervasi"
                ], key="clf_neigh_group")

            # -------- Otros inputs --------
            c3, c4, c5 = st.columns(3)
            with c3:
                clf_host_response_time = st.number_input("Host response time:", 0.0, 100.0,
                    float(mean_values.get("host_response_time", 0.0)),
                    key="clf_host_response_time")
                clf_host_response_rate = st.number_input("Host response rate (%):", 0.0, 100.0,
                    float(mean_values.get("host_response_rate", 0.0)),
                    key="clf_host_response_rate")
                clf_host_acceptance_rate = st.number_input("Host acceptance rate (%):", 0.0, 100.0,
                    float(mean_values.get("host_acceptance_rate", 0.0)),
                    key="clf_host_acceptance_rate")
            with c4:
                clf_host_is_superhost = st.selectbox("¿Es superhost?", [0, 1],
                    index=int(round(mean_values.get("host_is_superhost", 0))),
                    key="clf_host_is_superhost")
                clf_host_has_profile_pic = st.selectbox("¿Host tiene foto?", [0, 1],
                    index=int(round(mean_values.get("host_has_profile_pic", 1))),
                    key="clf_host_has_profile_pic")
                clf_host_identity_verified = st.selectbox("Identidad verificada:", [0, 1],
                    index=int(round(mean_values.get("host_identity_verified", 1))),
                    key="clf_host_identity_verified")
            with c5:
                clf_latitude = st.number_input("Latitud:", -90.0, 90.0,
                    float(mean_values.get("latitude", 0.0)),
                    key="clf_latitude")
                clf_longitude = st.number_input("Longitud:", -180.0, 180.0,
                    float(mean_values.get("longitude", 0.0)),
                    key="clf_longitude")
                clf_unused = st.number_input("Acomoda (duplicado UX):", 1, 16, 2,
                    key="clf_accommodates_dup")

            c6, c7, c8 = st.columns(3)
            with c6:
                clf_minimum_nights = st.number_input("Estancia mínima:", 1, 500,
                    int(mean_values.get("minimum_nights", 1)),
                    key="clf_minimum_nights")
                clf_maximum_nights = st.number_input("Estancia máxima:", 1, 500,
                    int(mean_values.get("maximum_nights", 30)),
                    key="clf_maximum_nights")
                clf_availability_30 = st.number_input("Disponibilidad 30 días:", 0, 30,
                    int(mean_values.get("availability_30", 0)),
                    key="clf_availability_30")

            with c7:
                clf_availability_365 = st.number_input("Disponibilidad 365 días:", 0, 365,
                    int(mean_values.get("availability_365", 0)),
                    key="clf_availability_365")
                clf_number_reviews = st.number_input("Número de reviews:", 0, 1000,
                    int(mean_values.get("number_of_reviews", 0)),
                    key="clf_number_reviews")
                clf_availability_eoy = st.number_input("Disponibilidad fin de año:", 0, 365,
                    int(mean_values.get("availability_eoy", 0)),
                    key="clf_availability_eoy")

            with c8:
                clf_estimated_occupancy = st.number_input("Ocupación 365 días:", 0.0, 100.0,
                    float(mean_values.get("estimated_occupancy_l365d", 0.0)),
                    key="clf_estimated_occupancy")
                clf_estimated_revenue = st.number_input("Ingresos 365 días:", 0.0, 500000.0,
                    float(mean_values.get("estimated_revenue_l365d", 0.0)),
                    key="clf_estimated_revenue")
                clf_reviews_per_month = st.number_input("Reviews por mes:", 0.0, 50.0,
                    float(mean_values.get("reviews_per_month", 0.0)),
                    key="clf_reviews_per_month")

            # -------- Review Scores --------
            c9, c10, c11 = st.columns(3)
            with c9:
                clf_score_rating = st.number_input("Rating:", 0.0, 5.0,
                    float(mean_values.get("review_scores_rating", 4.0)),
                    key="clf_score_rating")
                clf_score_accuracy = st.number_input("Accuracy:", 0.0, 5.0,
                    float(mean_values.get("review_scores_accuracy", 4.0)),
                    key="clf_score_accuracy")
                clf_score_cleanliness = st.number_input("Cleanliness:", 0.0, 5.0,
                    float(mean_values.get("review_scores_cleanliness", 4.0)),
                    key="clf_score_cleanliness")
                clf_neighbourhood_freq = st.number_input("Frecuencia barrio (%):",
                    0.0, 100.0,
                    float(mean_values.get("neighbourhood_cleansed_freq", 0.0)),
                    key="clf_neighbourhood_freq")
            with c10:
                clf_score_checkin = st.number_input("Check-in:", 0.0, 5.0,
                    float(mean_values.get("review_scores_checkin", 4.0)),
                    key="clf_score_checkin")
                clf_score_comm = st.number_input("Communication:", 0.0, 5.0,
                    float(mean_values.get("review_scores_communication", 4.0)),
                    key="clf_score_comm")
                clf_score_location = st.number_input("Location:", 0.0, 5.0,
                    float(mean_values.get("review_scores_location", 4.0)),
                    key="clf_score_location")
            with c11:
                clf_score_value = st.number_input("Value:", 0.0, 5.0,
                    float(mean_values.get("review_scores_value", 4.0)),
                    key="clf_score_value")
                clf_host_years = st.number_input("Años host:", 0, 50,
                    int(mean_values.get("host_years", 1)),
                    key="clf_host_years")
                clf_n_verifications = st.number_input("N° verificaciones host:", 0, 20,
                    int(mean_values.get("n_verifications", 0)),
                    key="clf_n_verifications")
                clf_calculated_host_listings_count = st.number_input("N° propiedades host:",
                    0, 1000,
                    int(mean_values.get("calculated_host_listings_count", 1)),
                    key="clf_calculated_host_listings_count")

            # ---------------- BOTÓN DE PREDICCIÓN ----------------
            if st.button("Clasificar Inmueble (TEC)"):
                ui = {
                    "host_response_time": clf_host_response_time,
                    "host_response_rate": clf_host_response_rate,
                    "host_acceptance_rate": clf_host_acceptance_rate,
                    "host_is_superhost": clf_host_is_superhost,
                    "host_has_profile_pic": clf_host_has_profile_pic,
                    "host_identity_verified": clf_host_identity_verified,
                    "latitude": clf_latitude,
                    "longitude": clf_longitude,
                    "accommodates": clf_accommodates,
                    "bathrooms": clf_bathrooms,
                    "bedrooms": clf_bedrooms,
                    "beds": clf_beds,
                    "minimum_nights": clf_minimum_nights,
                    "maximum_nights": clf_maximum_nights,
                    "availability_30": clf_availability_30,
                    "availability_365": clf_availability_365,
                    "number_of_reviews": clf_number_reviews,
                    "availability_eoy": clf_availability_eoy,
                    "estimated_occupancy_l365d": clf_estimated_occupancy,
                    "estimated_revenue_l365d": clf_estimated_revenue,
                    "review_scores_rating": clf_score_rating,
                    "review_scores_accuracy": clf_score_accuracy,
                    "review_scores_cleanliness": clf_score_cleanliness,
                    "review_scores_checkin": clf_score_checkin,
                    "review_scores_communication": clf_score_comm,
                    "review_scores_location": clf_score_location,
                    "review_scores_value": clf_score_value,
                    "instant_bookable": 1,
                    "calculated_host_listings_count": clf_calculated_host_listings_count,
                    "reviews_per_month": clf_reviews_per_month,
                    "n_amenities": clf_n_amenities,
                    "n_verifications": clf_n_verifications,
                    "host_years": clf_host_years,
                    "neighbourhood_cleansed_freq": clf_neighbourhood_freq,
                    "room_type": clf_room_type,
                    "neigh_group": clf_neigh_group
                }
                prob, clase = predict_clsf_tec(ui)
                st.success(f"Predicción tu inmueble es de: **{clase}**")
                st.info(f"La probabilidad estimada de exito de tu inmueble es de: **{prob:.2%}**")
            probs_all = []
            # df_30 = data.sample(frac=0.3, random_state=42)
            # for idx, row in df_30.iterrows():
            #     a=pd.DataFrame([row], columns=features)
            #     X_scaled = scaler_clf.transform(a)
            #     prob = float(model_clf.predict(X_scaled)[0][0])
            #     probs_all.append(prob)
            # df_30["probs_all"] = probs_all
            st.markdown("## Probabilidad de rentabilidad por zona y precio")
            barrios = ["Todos"] + sorted(df["neighbourhood_cleansed"].unique().tolist())
            selected_barrio = st.selectbox("Selecciona barrio:", barrios,
    key="select_barrio_principalp1")

            # Filtro por rango de precio
            min_price = int(df["price"].min())
            max_price = int(df["price"].max())
            price_range = st.slider("Rango de precio:", min_value=min_price, max_value=max_price,
                                    value=(min_price, max_price),key="price_range_slider_p1")

            # ---------- FILTRO DEL DATAFRAME ----------
            df_filtered = df.copy()

            # Filtra por barrio
            if selected_barrio != "Todos":
                df_filtered = df_filtered[df_filtered["neighbourhood_cleansed"] == selected_barrio]

            # Filtra por rango de precio
            df_filtered = df_filtered[(df_filtered["price"] >= price_range[0]) & 
                                    (df_filtered["price"] <= price_range[1])]

            # ---------- CÁLCULO DE RECOMENDADOS ----------
            q75_price = df_filtered["price"].quantile(0.75)
            df_filtered["recommended"] = np.where(
                (df_filtered["review_scores_rating"] >= 4.5) &
                (df_filtered["number_of_reviews"] >= 10) &
                (df_filtered["price"] <= q75_price),
                1,
                0
            )

            # ---------- GRÁFICO ----------
            fig_map_rent = px.scatter_mapbox(
                df_filtered,
                lat="latitude",
                lon="longitude",
                color="recommended",  
                size="price",  
                color_continuous_scale=px.colors.sequential.Viridis,
                size_max=15,
                zoom=12,
                mapbox_style="carto-positron",
 
                title="Propiedades Recomendadas por Ubicación"
            )

            st.plotly_chart(fig_map_rent, use_container_width=True)
        # ==========================================================
        # ---------------- CLASIFICADOR UNIANDES -------------------
        # ==========================================================
        with clf_tab2:
            st.subheader("Clasificador - Uniandes")

            st.write("Introduce las características del inmueble:")

            # -----------------------
            # INPUTS PRINCIPALES
            # -----------------------
            col1, col2 = st.columns(2)

            with col1:
                accommodates = st.number_input("Acomoda:", 1, 16, 2)
                bathrooms = st.number_input("Baños:", 0.0, 10.0, 1.0)
                bedrooms = st.number_input("Habitaciones:", 0, 10, 1)
                beds = st.number_input("Camas:", 1, 10, 1)
                n_amenities = st.number_input("Número de amenities:", 0, 50, 10)

            with col2:
                room_type = st.selectbox("Tipo habitación:", ["Hotel room", "Private room", "Shared room"])
                neigh_group = st.selectbox("Distrito:", [
                    "Eixample", "Gràcia", "Horta-Guinardó", "Les Corts",
                    "Nou Barris", "Sant Andreu", "Sant Martí",
                    "Sants-Montjuïc", "Sarrià-Sant Gervasi"
                ])
                property_type_clean = st.selectbox("Tipo de propiedad:", [
                    "Entire rental unit", "Entire serviced apartment", "Other",
                    "Private room in rental unit", "Room in hotel"
                ])


            # ================================================================
            # NUEVA SECCIÓN: Parámetros del host (valores iniciales = medias)
            # ================================================================
            st.markdown("---")
            st.markdown("### Parámetros del Host (Valores sugeridos del dataset)")

            colh1, colh2 = st.columns(2)

            with colh1:
                host_response_time = st.number_input(
                    "Tiempo de respuesta del host (horas)",
                    min_value=float(0.0), 
                    max_value=float(72.0),
                    value=float(mean_values["host_response_time"])
                )

                host_response_rate = st.number_input(
                    "Tasa de respuesta del host (0-1)",
                    min_value=float(0.0), 
                    max_value=float(100.0),
                    value=float(mean_values["host_response_rate"])
                )

                host_acceptance_rate = st.number_input(
                    "Tasa de aceptación (0-1)",
                    min_value=float(0.0), 
                    max_value=float(100.0),
                    value=float(mean_values["host_acceptance_rate"])
                )
                host_is_superhost = st.selectbox(
                    "¿Es Superhost?",
                    [0, 1],
                    index=int(round(mean_values["host_is_superhost"]))
                )
                host_has_profile_pic = st.selectbox(
                    "¿Tiene foto de perfil?",
                    [0, 1],
                    index=int(round(mean_values["host_has_profile_pic"]))
                )

            with colh2:
                host_identity_verified = st.selectbox(
                    "¿Identidad verificada?",
                    [0, 1],
                    index=int(round(mean_values["host_identity_verified"]))
                )
                minimum_nights = st.number_input(
                    "Mínimo de noches",
                    1, 400,
                    int(mean_values["minimum_nights"])
                )
                maximum_nights = st.number_input(
                    "Máximo de noches",
                    1, 9999,
                    int(mean_values["maximum_nights"])
                )
                calculated_host_listings_count = st.number_input(
                    "Número de propiedades del host",
                    0, 9999,
                    int(mean_values["calculated_host_listings_count"])
                )
                n_verifications = st.number_input(
                    "Número de verificaciones",
                    0, 10,
                    int(mean_values["n_verifications"])
                )

            st.markdown("### Ubicación y Frecuencia del Barrio")

            coll1, coll2 = st.columns(2)

            with coll1:
                latitude = st.number_input(
                    "Latitud",
                    -90.0, 90.0,
                    float(mean_values["latitude"])
                )
                host_years = st.number_input(
                    "Años del host en plataforma",
                    0, 20,
                    int(mean_values["host_years"])
                )

            with coll2:
                longitude = st.number_input(
                    "Longitud",
                    -180.0, 180.0,
                    float(mean_values["longitude"])
                )
                neighbourhood_cleansed_freq = st.number_input(
                    "Frecuencia relativa del barrio",
                    0.0, 0.2,
                    float(mean_values["neighbourhood_cleansed_freq"])
                )


            # ================================================================
            # BOTÓN Y PREDICCIÓN
            # ================================================================
            if st.button("Clasificar (Uniandes)"):

                # Inputs originales
                user_inputs = {
                    "accommodates": accommodates,
                    "bathrooms": bathrooms,
                    "bedrooms": bedrooms,
                    "beds": beds,
                    "n_amenities": n_amenities,
                    "room_type": room_type,
                    "neigh_group": neigh_group,
                    "property_type_clean": property_type_clean,
                }

                # Inputs añadidos
                user_inputs.update({
                    "host_response_time": host_response_time,
                    "host_response_rate": host_response_rate,
                    "host_acceptance_rate": host_acceptance_rate,
                    "host_is_superhost": host_is_superhost,
                    "host_has_profile_pic": host_has_profile_pic,
                    "host_identity_verified": host_identity_verified,
                    "latitude": latitude,
                    "longitude": longitude,
                    "minimum_nights": minimum_nights,
                    "maximum_nights": maximum_nights,
                    "instant_bookable": 1,
                    "calculated_host_listings_count": calculated_host_listings_count,
                    "n_verifications": n_verifications,
                    "host_years": host_years,
                    "neighbourhood_cleansed_freq": neighbourhood_cleansed_freq
                })

                # Predicción
                clase, prob = predecir_clasificacion(user_inputs)

                st.success(f"Predicción tu inmueble es de: **{clase}**")
                st.info(f"La probabilidad estimada de exito de tu inmueble es de: **{prob:.2%}**")
            # X = data[features]  
            # X_scaled = scaler_clf.transform(X)
            # probs_all = model_clf.predict(X_scaled)[:, 0] 
            probs_all = []
            # df_30 = data.sample(frac=0.3, random_state=42)
            # for idx, row in df_30.iterrows():
            #     a=pd.DataFrame([row], columns=features)
            #     X_scaled = scaler_clf.transform(a)
            #     prob = float(model_clf.predict(X_scaled)[0][0])
            #     probs_all.append(prob)
            # df_30["probs_all"] = probs_all
            st.markdown("## Probabilidad de rentabilidad por zona y precio")
            barrios = ["Todos"] + sorted(df["neighbourhood_cleansed"].unique().tolist())
            selected_barrio = st.selectbox("Selecciona barrio:", barrios)

            # Filtro por rango de precio
            min_price = int(df["price"].min())
            max_price = int(df["price"].max())
            price_range = st.slider("Rango de precio:", min_value=min_price, max_value=max_price,
                                    value=(min_price, max_price))

            # ---------- FILTRO DEL DATAFRAME ----------
            df_filtered = df.copy()

            # Filtra por barrio
            if selected_barrio != "Todos":
                df_filtered = df_filtered[df_filtered["neighbourhood_cleansed"] == selected_barrio]

            # Filtra por rango de precio
            df_filtered = df_filtered[(df_filtered["price"] >= price_range[0]) & 
                                    (df_filtered["price"] <= price_range[1])]

            # ---------- CÁLCULO DE RECOMENDADOS ----------
            q75_price = df_filtered["price"].quantile(0.75)
            df_filtered["recommended"] = np.where(
                (df_filtered["review_scores_rating"] >= 4.5) &
                (df_filtered["number_of_reviews"] >= 10) &
                (df_filtered["price"] <= q75_price),
                1,
                0
            )

            # ---------- GRÁFICO ----------
            fig_map_rent = px.scatter_mapbox(
                df_filtered,
                lat="latitude",
                lon="longitude",
                color="recommended",  
                size="price",  
                color_continuous_scale=px.colors.sequential.Viridis,
                size_max=15,
                zoom=12,
                mapbox_style="carto-positron",
 
                title="Propiedades Recomendadas por Ubicación"
            )

            st.plotly_chart(fig_map_rent, use_container_width=True,key="fig_map_rent_p2")
            
