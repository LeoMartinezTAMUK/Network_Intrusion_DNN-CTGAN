# Dataset used:Network Security Laboratory Knowledge Discovery in Databases (NSL-KDD)
# Citation: https://www.unb.ca/cic/datasets/nsl.html
# M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, “A Detailed Analysis of the KDD CUP 99 Data Set,”
# Submitted to Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.

# NSL-KDD 4-Class Network Intrusion Classification

# Written in Anaconda Spyder 5.5.0 IDE using Python 3.9.18 64-bit on Windows 10

#Imports

# CTGAN Imports
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

# Original Imports
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#%%
"""Load the datasets"""

# May need to change directory for dataset location ***

# --- Importing Train Dataset ---
# NSL-KDD, 43 features, 125973 samples, Multiclass Classification (From text file)
KDDTrain = pd.read_csv(r'C:Location/KDDTrain+.txt', header = None) # Data with difficulty level
# Column Headings
KDDTrain.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTrain.drop('difficulty', axis=1, inplace=True)

# --- Importing Test Dataset ---
# NSL-KDD, 43 features, 22544 samples, Multiclass Classification (From text file)
KDDTest = pd.read_csv(r'C:Location/KDDTest+.txt', header = None) # Data with difficulty level
# Column Headings
KDDTest.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class', 'difficulty']

# We will not utilize the 'difficulty' feature for now
KDDTest.drop('difficulty', axis=1, inplace=True)

#%%
"""Data Handling"""

# We drop 'num_outbound_cmds' from both training and testing dataset because every instance is equal to 0 in both datasets
KDDTrain.drop("num_outbound_cmds",axis=1,inplace=True)
KDDTest.drop("num_outbound_cmds",axis=1,inplace=True)

# We replace all instances with a value of 2 to 1 (1 instead of 0 because of Dr. Mishra's request) because the feature should be a binary value (0 or 1)
KDDTrain['su_attempted'] = KDDTrain['su_attempted'].replace(2, 1)
KDDTest['su_attempted'] = KDDTest['su_attempted'].replace(2, 1)

#%%
"""Class Assignment"""

# Distribution of attack classes in training dataset
KDDTrain['class'].value_counts()

# Distribution of attack classes in testing dataset
KDDTest['class'].value_counts()

#%%
# Change training attack labels to their respective attack class for multiclass classification
KDDTrain['class'].replace(['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land'],'DoS',inplace=True) # 6 sub classes of DoS
KDDTrain['class'].replace(['satan', 'ipsweep', 'portsweep', 'nmap'],'Probe',inplace=True) # 4 sub classes of Probe
KDDTrain['class'].replace(['warezclient', 'guess_passwd', 'warezmaster', 'imap', 'ftp_write', 'multihop', 'phf','spy'],'R2L',inplace=True) # 8 sub classes of R2L
KDDTrain['class'].replace(['buffer_overflow', 'rootkit', 'loadmodule','perl'],'U2R',inplace=True) # 4 sub classes of U2R

# Change testing attack labels to their respective attack class for multiclass classification
KDDTest['class'].replace(['neptune', 'apache2', 'processtable', 'smurf', 'back', 'mailbomb', 'pod', 'teardrop', 'land', 'udpstorm'],'DoS',inplace=True) # 10 sub classes of DoS
KDDTest['class'].replace(['mscan', 'satan', 'saint', 'portsweep', 'ipsweep', 'nmap'],'Probe',inplace=True) # 6 sub classes of Probe
KDDTest['class'].replace(['guess_passwd', 'warezmaster', 'snmpguess', 'snmpgetattack', 'httptunnel', 'multihop', 'named', 'sendmail', 'xlock', 'xsnoop', 'ftp_write', 'worm', 'phf', 'imap'],'R2L',inplace=True) # 14 sub classes of R2L
KDDTest['class'].replace(['buffer_overflow', 'ps', 'rootkit', 'xterm', 'loadmodule', 'perl', 'sqlattack'],'U2R',inplace=True) # 7 sub classes of U2R

#%%
"""Data Preprocessing"""

# Use LabelEncoding for categorical features (including 'class')

# Encode class label with LabelEncoder
label_encoder = preprocessing.LabelEncoder()
KDDTrain['class'] = label_encoder.fit_transform(KDDTrain['class'])
KDDTest['class'] = label_encoder.fit_transform(KDDTest['class'])

# Define the columns to LabelEncode
categorical_columns=['protocol_type', 'service', 'flag']

# Encode categorical columns using LabelEncoder
label_encoder = preprocessing.LabelEncoder()
for column in categorical_columns:
    KDDTrain[column] = label_encoder.fit_transform(KDDTrain[column])
    KDDTest[column] = label_encoder.transform(KDDTest[column])

# Define the columns to scale
columns_to_scale=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'count', 'srv_count', 'dst_host_count', 'dst_host_srv_count']

# Scale numerical columns using MinMax
scaler = MinMaxScaler()
for column in columns_to_scale:
    KDDTrain[column] = scaler.fit_transform(KDDTrain[[column]])
    KDDTest[column] = scaler.transform(KDDTest[[column]])

# Drop 'class' from X and make the Target Variable Y equal to 'class'
X_train = KDDTrain.iloc[:, :-1].values.astype('float32')
y_train = KDDTrain.iloc[:, -1].values
X_test = KDDTest.iloc[:, :-1].values.astype('float32')
y_test = KDDTest.iloc[:, -1].values

#%%
"""CTGAN (Conditional Tabular Generative Adversarial Network) Synthetic Data Creation"""
# Load data for synthetic data creation
data = KDDTrain

num_cols = ['duration','src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations',
       'num_shells','num_access_files', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate']

cat_cols = ['protocol_type', 'service', 'flag', 'class']

# Distribution of classes in dataset
data['class'].value_counts()

#%%
"""Define model and training parameters"""

# Defining the training parameters
batch_size = 5000
epochs = 100
learning_rate = 2e-4
beta_1 = 0.5
beta_2 = 0.9

ctgan_args = ModelParameters(batch_size=batch_size,
                             lr=learning_rate,
                             betas=(beta_1, beta_2))

train_args = TrainParameters(epochs=epochs)

#%%
"""Create and Train the CTGAN"""

print("Training In Progres...")

synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(data=data, train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

print("Training Complete!")
#%%
"""### Generate new synthetic data"""

synth_data = synth.sample(300000)
#print(synth_data)
#%%
"""Compare Real Samples to Synthetic"""

synth_data['class'].value_counts()

#%%
"""Drop Unneeded Synthetic Samples"""

# There is already a large sample size for class '4' and '0', no need for additional synthetic data
filtered_synth_data = synth_data[(synth_data['class'] != 4) & (synth_data['class'] != 0)] # Drop rows with class '4' or '0'

#%%
"""Concatenation of Synthetic Data with Real Data"""

# Assuming synth_data and KDDTrain are both pandas DataFrames
# If they are not, you can convert them to DataFrames using pd.DataFrame()

# Concatenate the synthetic samples to the original dataset
concatenated_data = pd.concat([KDDTrain, filtered_synth_data], ignore_index=True)

# Distribution of classes in dataset after synthetic concatenation
concatenated_data['class'].value_counts()

#%%
# Synthetic + Real Samples
X_train = concatenated_data.iloc[:, :-1].values.astype('float32')
y_train = concatenated_data.iloc[:, -1].values
X_test = KDDTest.iloc[:, :-1].values.astype('float32')
y_test = KDDTest.iloc[:, -1].values

#%%
"""Deep Neural Network for 4-class Classification (Softmax Regression)"""

# Deep Neural Network for 4 class classification

# Import necessary libraries
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as pyplot

# Class to drop 'normal' or '4' for 4 class classification
class_to_drop = 4

# Create a mask to filter out the samples belonging to the specified class
mask_train = y_train != class_to_drop
mask_test = y_test != class_to_drop

# Filter the data based on the mask
y_train_filtered = y_train[mask_train]
X_train_filtered = X_train[mask_train]

y_test_filtered = y_test[mask_test]
X_test_filtered = X_test[mask_test]

# Number of classes 0 = DoS, 1 = Probe, 2 = R2L, 3 = U2R | lexicographic order | 4 class classification
n_classes = 4
y_train_encoded = to_categorical(y_train_filtered, num_classes=n_classes)
y_test_encoded = to_categorical(y_test_filtered, num_classes=n_classes)

# Number of features in the input data (40 total features)
n_inputs = 40

# Define the input layer
visible = Input(shape=(n_inputs,))

# Hidden Layer 1
e = Dense(80, activation='relu')(visible)  # 80 neurons with ReLU activation

# Hidden layer 2
e = Dense(40, activation='relu')(e) # 40 neurons with ReLU activation

# Hidden Layer 3
e = Dense(4, activation='relu')(e) # 4 neurons with ReLU activation

# Output Layer
output = Dense(4, activation='softmax')(e) # Condensed to 4 neurons (for 4 classes)

# Define the autoencoder model
model = Model(inputs=visible, outputs=output)

# Cast the input data to float32
X_train_filtered = X_train_filtered.astype('float32')
X_test_filtered = X_test_filtered.astype('float32')

# Compile the model with a suitable loss function for classification, e.g., categorical cross-entropy
#model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Possible Better performance when a fixed learning rate is NOT used with Adam Optimizer, however not as stable/consistent overall
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping with a patience of 6 steps
early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

# Fit the autoencoder model to reconstruct input with batch size of 32 and 9 epochs
history = model.fit(X_train_filtered, y_train_encoded, epochs=25, batch_size=32, verbose=2, validation_split=0.15, callbacks=[early_stopping])

# Plot training loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.plot(history.history['accuracy'], label='train_accuracy')
pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()

# Define a deep network model
neural_network = Model(inputs=visible, outputs=output)
plot_model(neural_network, 'DNN_CTGAN.png', show_shapes=True)

# Save the neural_network model in Keras format
neural_network.save('DNN_CTGAN.keras')

#--------------------------------------------------------------------------------------------------------------------
# Data cleaning, preprocessing, sampling, and neural network has been applied prior to training
# SoftMax Regression Multiclass Classification (4 class) (For quick testing)

from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef

# Make predictions on the test data
y_pred = neural_network.predict(X_test_filtered)

# Convert the predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded true labels to class labels
y_test_classes = np.argmax(y_test_encoded, axis=1)

# Print classification report and confusion matrix on the test set
class_names = ["DoS", "Probe", "R2L", "U2R"]
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes), "\n")

# Calculate MCC
mcc_score = matthews_corrcoef(y_test_classes, y_pred_classes)
print("MCC Score:", mcc_score)

#%%
"""##Run Softmax Regression & Test Performance"""

# Performance comparison classifiers as evaluated via 10-fold cross-validation

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc
from keras.models import load_model
from sklearn.linear_model import LogisticRegression

# Load the model from file
encoder = load_model('DNN_CTGAN.keras')

# Encode the training and testing data
X_train_encoded = encoder.predict(X_train_filtered)
X_test_encoded = encoder.predict(X_test_filtered)

# Create a softmax regression model
smr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Perform 10-fold cross-validation
y_pred = cross_val_predict(smr, X_train_encoded, y_train_filtered, cv=10, n_jobs=-1)

outputFile=open(r'C:Location/Results_CTGAN.txt','a')
confusion= confusion_matrix(y_train_filtered, y_pred)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#Overall Accuracy
Accuracy_cla= accuracy_score(y_train_filtered, y_pred)
#Balanced Accuracy, Average of Recall/TPR and Efficiency/TNR
Bal_Accuracy_cla = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))
#Precision
Pr_cla=precision_score(y_train_filtered, y_pred, average='weighted')
#Sensivity/Recall/TPR
Sn_cla= recall_score(y_train_filtered, y_pred, average='weighted')
#F1 Score
F1_cla = f1_score(y_train_filtered, y_pred, average='weighted')
#Specificity/Efficiency/TNR
Sp_cla =(TN/float(TN+FP))
#Compute MCC
MCC_cla = matthews_corrcoef(y_train_filtered, y_pred)

# Did not use AUC for multiclass
#Compute auROC/AUC
#AUC_cla = roc_auc_score(y_train, y_pred, multi_class='ovr')

# False Positive Rate
FPR_cla = (FP/float(TN+FP))
#False Negative Rate (Miss Rate)
FNR_cla = (FN/float(FN+TP))

Results='TFIDF Cross Validation Results: \n'
outputFile.write(str(Results))
outputFile.write('TP = %.0f\n'%TP)
outputFile.write('FP = %.0f\n'%FP)
outputFile.write('TN = %.0f\n'%TN)
outputFile.write('FN = %.0f\n'%FN)
print("\nConfusion Matrix")
print("{0}".format(confusion_matrix(y_train_filtered, y_pred)))
outputFile.write('Overall Accuracy = %.5f\n'%Accuracy_cla)
outputFile.write('Balanced Accuracy = %.5f\n'%Bal_Accuracy_cla)
outputFile.write('Precision = %.5f\n'%Pr_cla)
outputFile.write('Sensivity/Recall/TPR = %.5f\n'%Sn_cla)
outputFile.write('F1 Score = %.5f\n'%F1_cla)
outputFile.write('Specificity/Efficiency/TNR = %.5f\n'%Sp_cla)
outputFile.write('MCC = %.5f\n'%MCC_cla)
#outputFile.write('auROC = %.5f\n'%AUC_cla)
print("\nClassification Report")
print("{0}".format(classification_report(y_train_filtered, y_pred)))

outputFile.close()

#%%
# Performance comparison classifiers as evaluated on independent test

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc
from keras.models import load_model
from sklearn.linear_model import LogisticRegression

# Load the model from file
encoder = load_model('DNN_CTGAN.keras')

# Encode the training and testing data
X_train_encoded = encoder.predict(X_train_filtered)
X_test_encoded = encoder.predict(X_test_filtered)

# Create a softmax regression model
smr = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# Train the model on the entire training set
smr.fit(X_train_encoded, y_train_filtered)

# Evaluate the model on the test set
y_pred = smr.predict(X_test_encoded)

outputFile=open(r'C:Location/Results_CTGAN.txt','a')
confusion= confusion_matrix(y_test_filtered, y_pred)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#Overall Accuracy
Accuracy_cla= accuracy_score(y_test_filtered, y_pred)
#Balanced Accuracy, Average of Recall/TPR and Efficiency/TNR
Bal_Accuracy_cla = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))
#Precision
Pr_cla=precision_score(y_test_filtered, y_pred, average='weighted')
#Sensivity/Recall/TPR
Sn_cla= recall_score(y_test_filtered, y_pred, average='weighted')
#F1 Score
F1_cla = f1_score(y_test_filtered, y_pred, average='weighted')
#Specificity/Efficiency/TNR
Sp_cla =(TN/float(TN+FP))
#Compute MCC
MCC_cla = matthews_corrcoef(y_test_filtered, y_pred)

# Did not use AUC for multiclass
#Compute auROC/AUC
#AUC_cla = roc_auc_score(y_test, y_pred, multi_class='ovr')

# False Positive Rate
FPR_cla = (FP/float(TN+FP))
#False Negative Rate (Miss Rate)
FNR_cla = (FN/float(FN+TP))

Results='Independent Test Results:\n'
outputFile.write(str(Results))
outputFile.write('TP = %.0f\n'%TP)
outputFile.write('FP = %.0f\n'%FP)
outputFile.write('TN = %.0f\n'%TN)
outputFile.write('FN = %.0f\n'%FN)
print("\nConfusion Matrix")
print("{0}".format(confusion_matrix(y_test_filtered, y_pred)))
outputFile.write('Overall Accuracy = %.5f\n'%Accuracy_cla)
outputFile.write('Balanced Accuracy = %.5f\n'%Bal_Accuracy_cla)
outputFile.write('Precision = %.5f\n'%Pr_cla)
outputFile.write('Sensivity/Recall/TPR = %.5f\n'%Sn_cla)
outputFile.write('F1 Score = %.5f\n'%F1_cla)
outputFile.write('Specificity/Efficiency/TNR = %.5f\n'%Sp_cla)
outputFile.write('MCC = %.5f\n'%MCC_cla)
#outputFile.write('auROC = %.5f\n'%AUC_cla)
print("\nClassification Report")
print("{0}".format(classification_report(y_test_filtered, y_pred)))

outputFile.close()