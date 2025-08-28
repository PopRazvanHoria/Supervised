"""
Project 1 — Vegetable Image Classification
This script is a linear export-friendly version of the notebook scaffold.
Set DATA_DIR to the local path where you downloaded the dataset (do NOT include data in submission).
"""

import os
import glob
import random
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

# Placeholder dataset path
DATA_DIR = 'path_to_downloaded_dataset'  # replace with actual path
IMAGE_SIZE = (128, 128)
RANDOM_SEED = 42


def load_image_paths(data_dir):
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    paths = []
    labels = []
    for i, cls in enumerate(sorted(classes)):
        cls_dir = os.path.join(data_dir, cls)
        files = glob.glob(os.path.join(cls_dir, '*'))
        for f in files:
            paths.append(f)
            labels.append(i)
    return paths, labels, classes


def extract_color_histogram(image_path, size=IMAGE_SIZE, bins=(8, 8, 8)):
    image = cv2.imread(image_path)
    if image is None:
        # return zeros if image cannot be read
        return np.zeros(np.prod(bins), dtype=np.float32)
    image = cv2.resize(image, size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_features_for_paths(paths):
    feats = [extract_color_histogram(p) for p in paths]
    return np.vstack(feats)


def build_simple_cnn(input_shape=(128, 128, 3), num_classes=10):
    # delayed import to avoid heavy dependency at module import time
    from tensorflow.keras import layers, models, optimizers
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_decision_tree(X_train, y_train):
    param_grid = {'max_depth': [5, 10, 20, None], 'min_samples_split': [2, 5, 10]}
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=1)
    clf.fit(X_train, y_train)
    return clf


def plot_cm(y_true, y_pred, labels, title='Confusion matrix'):
    import matplotlib.pyplot as plt
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap='Blues')
    plt.title(title)
    plt.show()


def main():
    print('This file is a scaffold. Edit DATA_DIR and run individual functions from an interactive session or integrate into a training script.')


if __name__ == '__main__':
    main()
