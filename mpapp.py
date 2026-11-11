
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Page configuration
st.set_page_config(page_title="MLP Classification", page_icon="🧠")

st.title("MLP Using Artificial Neural Networks")
st.write("Breast Cancer Classification using MLP")

# Load dataset
data = load_breast_cancer()
X = data.data
y = 1 - data.target

st.subheader("Dataset Information")

st.write("Total Instances:", X.shape[0])
st.write("Total Features:", X.shape[1])

st.dataframe(pd.DataFrame(X, columns=data.feature_names).head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Activation selection
activation = st.selectbox(
    "Select Hidden Layer Activation",
    ["relu", "tanh"]
)

# Build model
model = Sequential([
    Input(shape=(30,)),
    Dense(16, activation=activation),
    Dense(8, activation=activation),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

st.subheader("Model Architecture")

st.write("Input Layer: 30 neurons")
st.write("Hidden Layer 1: 16 neurons")
st.write("Hidden Layer 2: 8 neurons")
st.write("Output Layer: 1 neuron")

# Train button
if st.button("Train Model"):

    with st.spinner("Training model..."):

        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )

        predictions = model.predict(X_test, verbose=0)
        predictions = (predictions > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, predictions)

    st.success("Model Training Completed!")

    st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    st.write(pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"]
    ))

    st.subheader("Model Summary")

    st.write("Total Parameters:", model.count_params())
    st.write("Trainable Parameters:", model.count_params())
    st.write("Non-trainable Parameters: 0")

    st.write("Selected Activation:", activation)

st.info("MLP Binary Classification using Breast Cancer Dataset")
