from tensorflow.keras.models import load_model

# Load the model
model = load_model('modelA.h5')

# Summarize the model architecture
model.summary()
