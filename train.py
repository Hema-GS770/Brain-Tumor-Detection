from data_loader import load_data
from model import build_model
import os

print("Loading MRI dataset...")

# Load data
X, y = load_data()

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Build model
model = build_model()

print("Training started...")

history = model.fit(
    X,
    y,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    shuffle=True
)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model
model.save("models/segmentation_model.keras")

print("Training completed successfully!")
print("Model saved at: models/segmentation_model.keras")