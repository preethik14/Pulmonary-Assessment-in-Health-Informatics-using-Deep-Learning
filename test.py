from tensorflow.keras.models import load_model
import numpy as np
from utils.extract_features import extract_features
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model('LuCoNet.h5')

# Load and preprocess test file
file_path = '/home/rbccps/important files/luconet/Respiratory_Sound_Database/audio_and_txt_files/120_1b1_Pr_sc_Meditron.wav'
test_features = extract_features(file_path)
num_rows = 40
num_columns = 862
num_channels = 1
test_features = np.reshape(test_features, (1, num_rows, num_columns, num_channels))

# Prepare inputs
input_sound = test_features
input_disease = test_features

# Predict
predictions = model.predict([input_sound, input_disease])

sound_prediction = np.argmax(predictions[0], axis=1)[0]
disease_prediction = np.argmax(predictions[1], axis=1)[0]

print(f"Predicted sound label index: {sound_prediction}")
print(f"Predicted disease label index: {disease_prediction}")

# Load or define your label encoders
# (replace these with your actual encoders and classes)
le1 = LabelEncoder()
le1.classes_ = np.array(['Crackle', 'Healthy', 'Wheeze',  'Both'])  # example sound classes
le = LabelEncoder()
le.classes_ = np.array(['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI'])  # example disease classes

sound_class = le1.inverse_transform([sound_prediction])[0]
disease_class = le.inverse_transform([disease_prediction])[0]

print(f"Predicted sound label: {sound_class}")
print(f"Predicted disease label: {disease_class}")
