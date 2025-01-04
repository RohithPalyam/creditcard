import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from streamlit_option_menu import option_menu

# Streamlit app setup
st.title("Credit Card Fraud Detection Analysis")

# Option menu for tasks
with st.sidebar:
    selected = option_menu(
        menu_title="Tasks",
        options=[
            "Load Dataset",
            "Missing Value Analysis",
            "Transaction Statistics",
            "Visualization",
            "Normalize Amount",
            "Train-Test Split",
            "Train Models",
            "Compare Predictions",
            "Compare Accuracy",
            "Performance Matrix"
        ],
        default_index=0
    )

# Task 1: Load the dataset
if selected == "Load Dataset":
    with st.echo():
        data = pd.read_csv('creditcard.csv.crdownload')
    st.write("Dataset loaded successfully.")
    st.write(data.head())

# Task 2: Perform missing value analysis
if selected == "Missing Value Analysis":
    with st.echo():
        missing_values = data.isnull().sum()
    st.write("Missing values per column:", missing_values)

# Task 3: Calculate genuine and fraud transactions
if selected == "Transaction Statistics":
    with st.echo():
        genuine_count = data[data['Class'] == 0].shape[0]
        fraud_count = data[data['Class'] == 1].shape[0]
        total_transactions = data.shape[0]
        fraud_percentage = (fraud_count / total_transactions) * 100

    st.write(f"Number of genuine transactions: {genuine_count}")
    st.write(f"Number of fraud transactions: {fraud_count}")
    st.write(f"Percentage of fraud transactions: {fraud_percentage:.2f}%")

# Task 4: Visualize the genuine and fraudulent transactions
if selected == "Visualization":
    with st.echo():
        fig, ax = plt.subplots()
        ax.bar(['Genuine', 'Fraud'], [genuine_count, fraud_count], color=['blue', 'red'])
        ax.set_title('Transaction Counts')
        ax.set_ylabel('Count')
    st.pyplot(fig)

# Task 5: Normalize the 'Amount' column
if selected == "Normalize Amount":
    with st.echo():
        scaler = StandardScaler()
        data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
        data = data.drop(['Amount'], axis=1)
    st.write("Normalized 'Amount' column and dropped the original.")
    st.write(data.head())

# Task 6: Split the dataset into train and test sets
if selected == "Train-Test Split":
    with st.echo():
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write("Dataset split into training and testing sets.")

# Task 7: Train models
if selected == "Train Models":
    with st.echo():
        # Decision Tree
        decision_tree = DecisionTreeClassifier(random_state=42)
        decision_tree.fit(X_train, y_train)

        # Random Forest
        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(X_train, y_train)
    st.write("Models trained successfully.")

# Task 8: Compare predictions
if selected == "Compare Predictions":
    with st.echo():
        dt_predictions = decision_tree.predict(X_test)
        rf_predictions = random_forest.predict(X_test)
    st.write("Decision Tree Predictions:", dt_predictions[:10])
    st.write("Random Forest Predictions:", rf_predictions[:10])

# Task 9: Compare accuracy
if selected == "Compare Accuracy":
    with st.echo():
        dt_accuracy = accuracy_score(y_test, dt_predictions)
        rf_accuracy = accuracy_score(y_test, rf_predictions)
    st.write(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
    st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Task 10: Performance matrix comparison
if selected == "Performance Matrix":
    with st.echo():
        dt_report = classification_report(y_test, dt_predictions)
        rf_report = classification_report(y_test, rf_predictions)
        dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
        rf_conf_matrix = confusion_matrix(y_test, rf_predictions)
    st.write("Decision Tree Performance:\n", dt_report)
    st.write("Random Forest Performance:\n", rf_report)
    st.write("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
    st.write("Random Forest Confusion Matrix:\n", rf_conf_matrix)
