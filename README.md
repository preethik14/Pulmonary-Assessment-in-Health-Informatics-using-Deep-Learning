# Pulmonary-Assessment-in-Health-Informatics-using-Deep-Learning
## LuCoNet: Respiratory Sound and Disease Classification

LuCoNet is a deep learning model for classifying respiratory sounds into disease categories and sound labels using the **ICBHI 2017 Respiratory Sound Database**. This project aims to support research and applications in respiratory disease diagnosis using auscultation sound recordings.

---

## Dataset
Download the dataset from the official challenge website:  
  👉 [ICBHI 2017 Challenge Dataset](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)

## Getting Started
Follow these steps to set up and run the project.
1.Clone the Repository
```bash
git clone https://github.com/preethik14/luconet.git
cd luconet
```
2. Create a virtualenvironment and activate it
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install all the requirements
```bash
pip install -r requirements.txt
```
4. Extract the features and save it in features.npy file
```bash
python3 feature_extractor.py
```
5. Train using the extracted features. LuCoNet.h5 file is saved as output.
```bash
python3 train.py
```
6.Using the trained model inference can be drawn using test.py file
```bash
python3 test.py
```
## Acknowledgement 
We extend our thanks to many open-sourced works used in this project
1. [Multi-Task Learning](https://link.springer.com/article/10.1023/A:1007379606734)
2. [Respiratory Disease Classification](https://github.com/architgajpal/respiratory_disease_classification)

## Contact Us
Reach out to us to collaborate or for any questions!



