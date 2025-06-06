
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseConv2D, BatchNormalization, Dropout, ReLU, GlobalAveragePooling2D, Dense, concatenate
dropout_rate=0.2
filter_size = 2
def conv_block(x, filters, kernel_size=3, strides=1):
  
    x = DepthwiseConv2D(8, filter_size, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    return x