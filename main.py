import rasterio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================================
# STEP 1: Load RED and NIR bands
# ================================
red = rasterio.open('B04.jp2').read(1).astype(float)
nir = rasterio.open('B08.jp2').read(1).astype(float)

# ================================
# STEP 2: Calculate NDVI
# ================================
ndvi = (nir - red) / (nir + red + 1e-5)

# ================================
# STEP 3: Clean data
# ================================
ndvi = np.nan_to_num(ndvi)
ndvi = np.clip(ndvi, -1, 1)

# ================================
# STEP 4: Show NDVI map
# ================================
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map (Green = Healthy, Red = Poor)")
plt.show()

# ================================
# STEP 5: Create labels
# ================================
labels = np.where(ndvi > 0.4, 1, 0)

# ================================
# STEP 6: Prepare FULL dataset
# ================================
X_full = ndvi.reshape(-1, 1)
y_full = labels.reshape(-1)

# ================================
# STEP 7: Sampling (for fast ML)
# ================================
sample_size = 5000
indices = np.random.choice(len(X_full), sample_size, replace=False)

X_sample = X_full[indices]
y_sample = y_full[indices]

# ================================
# STEP 8: Train-test split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42
)

# ================================
# STEP 9: Train model
# ================================
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# ================================
# STEP 10: Accuracy
# ================================
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# ================================
# STEP 11: Predict FULL image
# ================================
prediction_full = model.predict(X_full)
prediction_map = prediction_full.reshape(ndvi.shape)

# ================================
# STEP 12: Show prediction map
# ================================
plt.imshow(prediction_map, cmap='RdYlGn')
plt.title("Crop Health Prediction (ML Output)")
plt.show()

# ================================
# STEP 13: Save outputs
# ================================
plt.imsave("ndvi_output.png", ndvi, cmap='RdYlGn')
plt.imsave("prediction_output.png", prediction_map, cmap='RdYlGn')

print("✅ Project completed! Images saved.")