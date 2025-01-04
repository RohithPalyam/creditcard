import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
st.subheader("Load Dataset")
with st.echo():
    data = pd.read_csv('creditcard.csv.crdownload')
    st.write("Dataset loaded successfully.")
    st.write(data.head())

# Analyze missing values
st.subheader("Analyze Missing Values")
with st.echo():
    missing_values = data.isnull().sum()
    st.write("Missing values per column:", missing_values)

# Calculate statistics
st.subheader("Calculate Statistics")
with st.echo():
    genuine_count = data[data['Class'] == 0].shape[0]
    fraud_count = data[data['Class'] == 1].shape[0]
    total_transactions = data.shape[0]
    fraud_percentage = (fraud_count / total_transactions) * 100

    st.write(f"Number of genuine transactions: {genuine_count}")
    st.write(f"Number of fraud transactions: {fraud_count}")
    st.write(f"Percentage of fraud transactions: {fraud_percentage:.2f}%")

# Visualize transactions
st.subheader("Visualize Transactions")
with st.echo():
    fig, ax = plt.subplots()
    ax.bar(['Genuine', 'Fraud'], [genuine_count, fraud_count], color=['blue', 'red'])
    ax.set_title('Transaction Counts')
    ax.set_ylabel('Count')

    st.pyplot(fig)

# Normalize 'Amount' column
st.subheader("Normalize 'Amount' Column")
with st.echo():
    scaler = StandardScaler()
    data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Amount'], axis=1)
    st.write("Normalized 'Amount' column and dropped the original.")
    st.write(data.head())

# Split dataset into training and testing sets
st.subheader("Split Dataset")
with st.echo():
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write("Dataset split into training and testing sets.")

# Train models
st.subheader("Train Models")
with st.echo():
    # Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)

    # Random Forest
    random_forest = RandomForestClassifier(random_state=42)
    random_forest.fit(X_train, y_train)

    st.write("Models trained successfully.")

# Compare predictions
st.subheader("Compare Predictions")
with st.echo():
    dt_predictions = decision_tree.predict(X_test)
    rf_predictions = random_forest.predict(X_test)

    st.write("Decision Tree Predictions:", dt_predictions[:10])
    st.write("Random Forest Predictions:", rf_predictions[:10])

# Compare accuracy
st.subheader("Compare Accuracy")
with st.echo():
    # Debug shapes and missing values
    st.write(f"y_test shape: {y_test.shape}")
    st.write(f"Decision Tree Predictions shape: {dt_predictions.shape}")
    st.write(f"Missing values in y_test: {pd.isnull(y_test).sum()}")

    # Ensure y_test and predictions are valid
    if len(y_test) == 0 or len(dt_predictions) == 0 or pd.isnull(y_test).any():
        st.error("y_test or predictions contain invalid values or are empty.")
    else:
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        rf_accuracy = accuracy_score(y_test, rf_predictions)

        st.write(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
        st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")


# Performance matrix
st.subheader("Performance Matrix")
with st.echo():
    dt_report = classification_report(y_test, dt_predictions)
    rf_report = classification_report(y_test, rf_predictions)
    dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
    rf_conf_matrix = confusion_matrix(y_test, rf_predictions)

    st.write("Decision Tree Performance:\n", dt_report)
    st.write("Random Forest Performance:\n", rf_report)
    st.write("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
    st.write("Random Forest Confusion Matrix:\n", rf_conf_matrix)
