# syntax=docker/dockerfile:1
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py modelo_clasificacion.keras scaler_clasificacion.pkl data_clean_barcelona.csv /app/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]


# # syntax=docker/dockerfile:1
# FROM ubuntu:22.04

# # ----------------------------
# # 1. Install base dependencies
# # ----------------------------
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev \
#     build-essential \
#     && apt-get clean

# # ----------------------------
# # 2. Install Python libraries
# # ----------------------------
# # TensorFlow CPU is heavy: install it separately to avoid timeout or conflicts
# RUN pip install --upgrade pip
# RUN pip install tensorflow-cpu==2.15.0

# # Core dependencies
# RUN pip install --no-cache-dir \
#     # tensorflow-cpu==2.15.0 \
#     # keras==3.0.5 \
#     h5py==3.10.0 \
#     numpy==1.24.3 \
#     joblib \
#     streamlit \
#     pandas \
#     plotly

# # Install TensorFlow CPU (MUY IMPORTANTE: evita problemas de GPU)

# # ----------------------------
# # 3. Copy your application
# # ----------------------------
# WORKDIR /app

# # Copy everything into the container
# COPY app.py modelo_clasificacion.keras scaler_clasificacion.pkl data_clean_barcelona.csv /app/

# # ----------------------------
# # 4. Expose Streamlit port
# # ----------------------------
# EXPOSE 8501

# # ----------------------------
# # 5. Run the Streamlit app
# # ----------------------------
# # CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
# CMD ["streamlit", "run", "app.py"]
