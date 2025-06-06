from sklearn.model_selection import train_test_split
from keras.layers import Input
from utils.conv_block import conv_block
from feature_extractor import features1, features2, oh_labels, oh_labels1
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from datetime import datetime
x_train_disease, x_test_disease, y_train_disease, y_test_disease = train_test_split(features1, oh_labels, stratify=oh_labels, 
                                                    test_size=0.2, random_state = 42)
x_train_sound, x_test_sound, y_train_sound, y_test_sound = train_test_split(features2, oh_labels1, stratify=oh_labels1, 
                                                    test_size=0.2, random_state = 42)

num_rows = 40
num_columns = 862
num_channels = 1
num_labels_disease = 6
num_labels_sound = 4

filter_size = 2

input_shape_sound = (num_rows, num_columns, num_channels)  # Adjust based on your sound data shape


input_shape_disease = (num_rows, num_columns, num_channels)  # Adjust based on your disease data shape


filter_size = 2

input_sound = Input(shape=input_shape_sound, name='input_sound')
input_disease = Input(shape=input_shape_disease, name='input_disease')
dropout_rate=0.2
merged = concatenate([input_sound, input_disease])

x1 = Conv2D(16, kernel_size=3, strides=2, padding='same')(merged)
x1 = conv_block(x1, filters=32)
x1 = conv_block(x1, filters=64)
x1 = conv_block(x1, filters=128)
x1 = conv_block(x1, filters=256)
global_avg_pooling = GlobalAveragePooling2D()(x1)
dense_merged = Dense(256, activation='relu')(global_avg_pooling)
output_sound = Dense(num_labels_sound, activation='softmax', name='sound_output')(dense_merged)
output_disease = Dense(num_labels_disease, activation='softmax', name='disease_output')(dense_merged)
model = Model(inputs=[input_sound, input_disease], outputs=[output_sound, output_disease])


# Compile the model
model.compile(optimizer='adam', loss={'sound_output': 'categorical_crossentropy', 'disease_output': 'categorical_crossentropy'}, metrics={'sound_output' : 'accuracy', 'disease_output' : 'accuracy'})

start = datetime.now()

history = model.fit(
    [x_train_sound, x_train_disease],  # Input features for sound and disease
    {'sound_output': y_train_sound, 'disease_output': y_train_disease},  # Output labels for sound and disease
    epochs=20,
    batch_size=16,
    validation_split=0.2,
)
duration = datetime.now() - start

print("Training completed in time: ", duration)
model.save('LuCoNet.h5')
print("Model saved successfully as 'LuCoNet.h5'")
