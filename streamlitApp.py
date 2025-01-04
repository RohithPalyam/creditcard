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
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.decision_tree = None
        self.random_forest = None

    def load_data(self):
        try:
            self.data = pd.read_csv("creditcard.csv.crdownload")
            st.write("Dataset loaded successfully!")
            return self.data
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    def analyze_missing_values(self):
        return self.data.isnull().sum()

    def calculate_statistics(self):
        genuine_count = self.data[self.data['Class'] == 0].shape[0]
        fraud_count = self.data[self.data['Class'] == 1].shape[0]
        total_transactions = self.data.shape[0]
        fraud_percentage = (fraud_count / total_transactions) * 100
        return genuine_count, fraud_count, fraud_percentage

    def visualize_transactions(self, genuine_count, fraud_count):
        fig, ax = plt.subplots()
        ax.bar(['Genuine', 'Fraud'], [genuine_count, fraud_count], color=['blue', 'red'])
        ax.set_title('Transaction Counts')
        ax.set_ylabel('Count')
        return fig

    def normalize_amount(self):
        scaler = StandardScaler()
        self.data['NormalizedAmount'] = scaler.fit_transform(self.data['Amount'].values.reshape(-1, 1))
        self.data = self.data.drop(['Amount'], axis=1)
        return self.data

    def split_data(self):
        X = self.data.drop('Class', axis=1)
        y = self.data['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def train_models(self):
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.decision_tree.fit(self.X_train, self.y_train)

        self.random_forest = RandomForestClassifier(random_state=42)
        self.random_forest.fit(self.X_train, self.y_train)

    def get_predictions(self):
        dt_predictions = self.decision_tree.predict(self.X_test)
        rf_predictions = self.random_forest.predict(self.X_test)
        return dt_predictions, rf_predictions

    def calculate_accuracy(self, dt_predictions, rf_predictions):
        dt_accuracy = accuracy_score(self.y_test, dt_predictions)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        return dt_accuracy, rf_accuracy

    def performance_matrix(self, dt_predictions, rf_predictions):
        dt_report = classification_report(self.y_test, dt_predictions)
        rf_report = classification_report(self.y_test, rf_predictions)
        dt_conf_matrix = confusion_matrix(self.y_test, dt_predictions)
        rf_conf_matrix = confusion_matrix(self.y_test, rf_predictions)
        return dt_report, rf_report, dt_conf_matrix, rf_conf_matrix


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

# Task 1: Load the dataset
if selected == "Load Dataset":
    data = app.load_data()

# Task 2: Perform missing value analysis
if selected == "Missing Value Analysis" and app.data is not None:
    missing_values = app.analyze_missing_values()
    st.write("Missing values per column:", missing_values)

# Task 3: Calculate genuine and fraud transactions
if selected == "Transaction Statistics" and app.data is not None:
    genuine_count, fraud_count, fraud_percentage = app.calculate_statistics()
    st.write(f"Number of genuine transactions: {genuine_count}")
    st.write(f"Number of fraud transactions: {fraud_count}")
    st.write(f"Percentage of fraud transactions: {fraud_percentage:.2f}%")

# Task 4: Visualize the genuine and fraudulent transactions
if selected == "Visualization" and app.data is not None:
    fig = app.visualize_transactions(genuine_count, fraud_count)
    st.pyplot(fig)

# Task 5: Normalize the 'Amount' column
if selected == "Normalize Amount" and app.data is not None:
    normalized_data = app.normalize_amount()
    st.write("Normalized 'Amount' column and dropped the original.")
    st.write(normalized_data.head())

# Task 6: Split the dataset into train and test sets
if selected == "Train-Test Split" and app.data is not None:
    app.split_data()
    st.write("Dataset split into training and testing sets.")

# Task 7: Train models
if selected == "Train Models" and app.X_train is not None and app.X_test is not None:
    app.train_models()
    st.write("Models trained successfully.")

# Task 8: Compare predictions
if selected == "Compare Predictions" and app.decision_tree and app.random_forest:
    dt_predictions, rf_predictions = app.get_predictions()
    st.write("Decision Tree Predictions:", dt_predictions[:10])
    st.write("Random Forest Predictions:", rf_predictions[:10])

# Task 9: Compare accuracy
if selected == "Compare Accuracy" and app.decision_tree and app.random_forest:
    dt_predictions, rf_predictions = app.get_predictions()
    dt_accuracy, rf_accuracy = app.calculate_accuracy(dt_predictions, rf_predictions)
    st.write(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
    st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Task 10: Performance matrix comparison
if selected == "Performance Matrix" and app.decision_tree and app.random_forest:
    dt_predictions, rf_predictions = app.get_predictions()
    dt_report, rf_report, dt_conf_matrix, rf_conf_matrix = app.performance_matrix(dt_predictions, rf_predictions)
    st.write("Decision Tree Performance:\n", dt_report)
    st.write("Random Forest Performance:\n", rf_report)
    st.write("Decision Tree Confusion Matrix:\n", dt_conf_matrix)
    st.write("Random Forest Confusion Matrix:\n", rf_conf_matrix)
