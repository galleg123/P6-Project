import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, fbeta_score, make_scorer
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

# Creating a function that uses category_id to determine if a cage is present.
def cage_detection(row):
	if (row['category_id'] == 0) or (row['category_id'] == 1):
		return 1
	else:
		return 0

# Defining the different columns and features
columns = ['video_id','path','width','height','fps','total_frames','file_name','category_id','category_name','supercategory','color','metadata', 'pass_id','Frame','BoudingBox','Area','Circularity','Convexity','Rectangularity','Elongation','Eccentricity','Solidity']
features = ['Area', 'Circularity', 'Convexity', 'Rectangularity', 'Elongation', 'Eccentricity', 'Solidity']

# Import csv with all data
df = pd.read_csv('featuresExtracted_no_touchy_noDoubleFrames.csv', usecols=columns)

# Make a new cage column that applies the function to each row
df['Cage'] = df.apply (lambda row: cage_detection(row), axis=1)

# Make a dataframe containing all features
X = df[features]

# Make a column with results of the classification
y = df['Cage']

# Prepare test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Weighted_SVM = pickle.load(open('classes/model/finalized_model.sav', 'rb')) # Load in the model from the disk
Scaler = pickle.load(open('classes/model/finalized_scaler.sav', 'rb')) # Load in the model from the disk

# Scale test data
X_train_scaled = Scaler.transform(X_train)
X_test_scaled = Scaler.transform(X_test)

pred_test = Weighted_SVM.predict(X_test_scaled)
f4_score = f'F4 Score = {fbeta_score(y_test, pred_test, beta=4)}'
print(f4_score)

cm = confusion_matrix(y_test, pred_test, labels=Weighted_SVM.classes_)

# Display new confusion matrix
after = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Is not a cage","Is a cage"])
after.plot()
plt.show()