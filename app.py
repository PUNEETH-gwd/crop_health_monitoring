import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ================================
# Title
# ================================
st.title("🌾 Crop Health Monitoring (NDVI + ML)")
st.write("Upload Sentinel-2 B04 (Red) and B08 (NIR) files")

# ================================
# File Upload
# ================================
red_file = st.file_uploader("Upload B04 (Red band)", type=["jp2"])
nir_file = st.file_uploader("Upload B08 (NIR band)", type=["jp2"])

# ================================
# Process when files uploaded
# ================================
if red_file and nir_file:

    st.success("Files uploaded successfully!")

    # Save temp files
    with open("temp_red.jp2", "wb") as f:
        f.write(red_file.read())

    with open("temp_nir.jp2", "wb") as f:
        f.write(nir_file.read())

    # ================================
    # Read Bands
    # ================================
    red = rasterio.open("temp_red.jp2").read(1).astype(float)
    nir = rasterio.open("temp_nir.jp2").read(1).astype(float)

    # ================================
    # NDVI Calculation
    # ================================
    ndvi = (nir - red) / (nir + red + 1e-5)
    ndvi = np.nan_to_num(ndvi)
    ndvi = np.clip(ndvi, -1, 1)

    # ================================
    # Show NDVI Map
    # ================================
    st.subheader("🌱 NDVI Map")

    fig1, ax1 = plt.subplots()
    cax1 = ax1.imshow(ndvi, cmap='RdYlGn')
    fig1.colorbar(cax1)
    st.pyplot(fig1)

    # ================================
    # Crop Health Analysis
    # ================================
    total_pixels = ndvi.size

    healthy = np.sum(ndvi > 0.6)
    moderate = np.sum((ndvi > 0.3) & (ndvi <= 0.6))
    poor = np.sum(ndvi <= 0.3)

    healthy_pct = (healthy / total_pixels) * 100
    moderate_pct = (moderate / total_pixels) * 100
    poor_pct = (poor / total_pixels) * 100

    st.subheader("🌾 Crop Health Analysis")

    st.write(f"🌱 Healthy: {healthy_pct:.2f}%")
    st.write(f"🟡 Moderate: {moderate_pct:.2f}%")
    st.write(f"🔴 Poor: {poor_pct:.2f}%")

    # Final interpretation
    if healthy_pct > 70:
        st.success("✅ Crops are mostly healthy")
    elif moderate_pct > 50:
        st.warning("⚠️ Crops are moderately healthy")
    else:
        st.error("❌ Crops are in poor condition")

    # ================================
    # Machine Learning
    # ================================
    X_full = ndvi.reshape(-1, 1)
    y_full = np.where(ndvi > 0.4, 1, 0).reshape(-1)

    # Sampling for speed
    sample_size = 5000
    indices = np.random.choice(len(X_full), sample_size, replace=False)

    X_sample = X_full[indices]
    y_sample = y_full[indices]

    # Train model
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_sample, y_sample)

    # Predict full image
    prediction_full = model.predict(X_full)
    prediction_map = prediction_full.reshape(ndvi.shape)

    # ================================
    # Show Prediction Map
    # ================================
    st.subheader("🗺️ ML Crop Health Prediction")

    fig2, ax2 = plt.subplots()
    cax2 = ax2.imshow(prediction_map, cmap='RdYlGn')
    fig2.colorbar(cax2)
    st.pyplot(fig2)

    # ================================
    # Accuracy
    # ================================
    acc = model.score(X_sample, y_sample)
    st.write(f"📊 Model Accuracy: {acc:.3f}")

    st.success("✅ Processing Complete!")