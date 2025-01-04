import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from streamlit_option_menu import option_menu

class FraudDetectionApp:
    def __init__(self):
        pass

    def load_data(self):
        with st.echo():
            self.data = pd.read_csv('creditcard.csv.crdownload')
            st.write("Dataset loaded successfully.")
            st.write(self.data.head())

    def analyze_missing_values(self):
        with st.echo():
            missing_values = self.data.isnull().sum()
            st.write("Missing values per column:", missing_values)

    def calculate_statistics(self):
        with st.echo():
            genuine_count = self.data[self.data['Class'] == 0].shape[0]
            fraud_count = self.data[self.data['Class'] == 1].shape[0]
            total_transactions = self.data.shape[0]
            fraud_percentage = (fraud_count / total_transactions) * 100

            st.write(f"Number of genuine transactions: {genuine_count}")
            st.write(f"Number of fraud transactions: {fraud_count}")
            st.write(f"Percentage of fraud transactions: {fraud_percentage:.2f}%")

    def visualize_transactions(self):
        with st.echo():
            genuine_count = self.data[self.data['Class'] == 0].shape[0]
            fraud_count = self.data[self.data['Class'] == 1].shape[0]

            fig, ax = plt.subplots()
            ax.bar(['Genuine', 'Fraud'], [genuine_count, fraud_count], color=['blue', 'red'])
            ax.set_title('Transaction Counts')
            ax.set_ylabel('Count')

            st.pyplot(fig)

    def normalize_amount(self):
        with st.echo():
            scaler = StandardScaler()
            self.data['NormalizedAmount'] = scaler.fit_transform(self.data['Amount'].values.reshape(-1, 1))
            self.data = self.data.drop(['Amount'], axis=1)
            st.write("Normalized 'Amount' column and dropped the original.")
            st.write(self.data.head())

    def split_data(self):
        with st.echo():
            X = self.data.drop('Class', axis=1)
            y = self.data['Class']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.write("Dataset split into training and testing sets.")

    def train_models(self):
        with st.echo():
            # Decision Tree
            self.decision_tree = DecisionTreeClassifier(random_state=42)
            self.decision_tree.fit(self.X_train, self.y_train)

            # Random Forest
            self.random_forest = RandomForestClassifier(random_state=42)
            self.random_forest.fit(self.X_train, self.y_train)

            st.write("Models trained successfully.")

    def compare_predictions(self):
        with st.echo():
            dt_predictions = self.decision_tree.predict(self.X_test)
            rf_predictions = self.random_forest.predict(self.X_test)
            st.write("Decision Tree Predictions:", dt_predictions[:10])
            st.write("Random Forest Predictions:", rf_predictions[:10])

    def compare_accuracy(self):
        with st.echo():
            dt_predictions = self.decision_tree.predict(self.X_test)
            rf_predictions = self.random_forest.predict(self.X_test)

            dt_accuracy = accuracy_score(self.y_test, dt_predictions)
            rf_accuracy = accuracy_score(self.y_test, rf_predictions)

            st.write(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
            st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

    def performance_matrix(self):
        with st.echo():
            dt_predictions = self.decision_tree.predict(self.X_test)
            rf_predictions = self.random_forest.predict(self.X_test)

            dt_report = classification_report(self.y_test, dt_predictions)
            rf_report = classification_report(self.y_test, rf_predictions)
            dt_conf_matrix = confusion_matrix(self.y_test, dt_predictions)
            rf_conf_matrix = confusion_matrix(self.y_test, rf_predictions)

            st.write("Decision Tree Performance:\n", dt_report)
            st.write("Random Forest Performance:\n", rf_report)
            st.write("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
            st.write("Random Forest Confusion Matrix:\n", rf_conf_matrix)

# Streamlit app setup
st.title("Credit Card Fraud Detection Analysis")
app = FraudDetectionApp()

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

# Task execution
if selected == "Load Dataset":
    app.load_data()

if selected == "Missing Value Analysis":
    app.analyze_missing_values()

if selected == "Transaction Statistics":
    app.calculate_statistics()

if selected == "Visualization":
    app.visualize_transactions()

if selected == "Normalize Amount":
    app.normalize_amount()

if selected == "Train-Test Split":
    app.split_data()

if selected == "Train Models":
    app.train_models()

if selected == "Compare Predictions":
    app.compare_predictions()

if selected == "Compare Accuracy":
    app.compare_accuracy()

if selected == "Performance Matrix":
    app.performance_matrix()
