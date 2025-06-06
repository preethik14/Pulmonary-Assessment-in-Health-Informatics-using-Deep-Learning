from datetime import datetime
from os import listdir
from os.path import isfile, join
import librosa
import librosa.display
import os
import numpy as np
import pandas as pd
from utils.extract_features import extract_features
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


mypath = "/home/rbccps/important files/luconet/Respiratory_Sound_Database/audio_and_txt_files"
filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))] 
p_id_in_file = [] # patient IDs corresponding to each file
for name in filenames:
    p_id_in_file.append(int(name[:3]))

p_id_in_file = np.array(p_id_in_file) 
print(p_id_in_file)

max_pad_len = 862 

filepaths = [os.path.join(mypath, f) for f in filenames]


p_diag1 = pd.read_csv("/home/rbccps/important files/luconet/Respiratory_Sound_Database/patient_info.csv",header=None)

disease_labels = np.array([p_diag1[p_diag1[0] == x][1].values[0] for x in p_id_in_file])
sound_labels = np.array([p_diag1[p_diag1[0] == x][2].values[0] for x in p_id_in_file])
features = [] 

for file_name in filepaths:
    data = extract_features(file_name)
    features.append(data)

features = np.array(features)
np.save("features.npy", features)

le = LabelEncoder()
i_labels = le.fit_transform(disease_labels)
le1 = LabelEncoder()
i_labels1 = le1.fit_transform(sound_labels)
oh_labels = to_categorical(i_labels)
oh_labels1 = to_categorical(i_labels1)

features1 = np.reshape(features, (*features.shape,1)) 
features2 = np.reshape(features, (*features.shape,1)) 


