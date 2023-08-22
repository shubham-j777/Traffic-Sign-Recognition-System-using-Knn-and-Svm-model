import os   #To load Files
import pandas as pd     #To import csv files for image path and labels
import numpy as np      #To create array of Images
from skimage.io import imread   #to read images from the path
from skimage.color import rgb2gray       #to greyscale images
from skimage.transform import resize       #to resize all images to a specific format
from skimage.feature import hog     #To extract features from images
from sklearn.neighbors import KNeighborsClassifier      #KNN Model
from sklearn.svm import SVC         #SVM Model
from sklearn.model_selection import train_test_split    #splitting training and testing data
import joblib       #to export model to pkl file
from sklearn.metrics import classification_report #for classification report of the model

# Path to the directory containing the GTSRB dataset
data_dir = "D:/college files/DS course/trafic recog project code/GTSRB/Train"

# Path to the CSV file containing image paths and class IDs
csv_file_path = "D:/college files/DS course/trafic recog project code/GTSRB/Train.csv"

# Load the CSV file into a DataFrame
data_df = pd.read_csv(csv_file_path)

# Initialize empty lists to store data and labels
data = []
labels = []

# Load images and labels from the CSV file
for _, row in data_df.iterrows():
    image_path = os.path.join(data_dir, row['Path'][6:])  # Remove "Train/" from the path
    image = imread(image_path)
    image_gray = rgb2gray(image)
    image_resized = resize(image_gray, (32, 32))
    data.append(image_resized)
    labels.append(row['ClassId'])

# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Feature extraction using HOG
def extract_hog_features(images):
    hog_features = []
    for image in images:
        features = hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(features)
    return np.array(hog_features)

X_features = extract_hog_features(data)

# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.30, random_state=42)

# Initialize k-NN model
knn_model = KNeighborsClassifier(n_neighbors=1)  # Choose k value

# Train the k-NN model
knn_model.fit(X_train, y_train)

# Evaluate k-NN model accuracy
knn_accuracy = knn_model.score(X_test, y_test)
print("k-NN Model Accuracy:", knn_accuracy)

# Save the trained k-NN model to a file
knn_model_file_path = "D:/college files/DS course/trafic recog project code/GTSRB/traffic_sign_knn_model.pkl"
joblib.dump(knn_model, knn_model_file_path)

y_pred = knn_model.predict(X_test)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# Initialize SVM model
svm_model = SVC(kernel='rbf', C=10)  # Linear kernel, adjust C value if needed

# Train the SVM model
svm_model.fit(X_train, y_train)

# Evaluate SVM model accuracy
svm_accuracy = svm_model.score(X_test, y_test)
print("SVM Model Accuracy:", svm_accuracy)

# Save the trained SVM model to a file
svm_model_file_path = "D:/college files/DS course/trafic recog project code/GTSRB/traffic_sign_svm_model.pkl"
joblib.dump(svm_model, svm_model_file_path)

y_pred = svm_model.predict(X_test)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)
