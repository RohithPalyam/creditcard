import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('/mnt/data/creditcard.csv.crdownload')

# Task 2: Perform missing value analysis
missing_values = data.isnull().sum()
print("Missing values per column:\n", missing_values)

# Task 3: Calculate genuine and fraud transactions
genuine_count = data[data['Class'] == 0].shape[0]
fraud_count = data[data['Class'] == 1].shape[0]
total_transactions = data.shape[0]
fraud_percentage = (fraud_count / total_transactions) * 100

print(f"Number of genuine transactions: {genuine_count}")
print(f"Number of fraud transactions: {fraud_count}")
print(f"Percentage of fraud transactions: {fraud_percentage:.2f}%")

# Task 4: Visualize the genuine and fraudulent transactions
plt.bar(['Genuine', 'Fraud'], [genuine_count, fraud_count], color=['blue', 'red'])
plt.title('Transaction Counts')
plt.ylabel('Count')
plt.show()

# Task 5: Normalize the 'Amount' column
scaler = StandardScaler()
data['NormalizedAmount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Amount'], axis=1)

# Task 6: Split the dataset into train and test sets
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Task 7: Train models
# Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

dt_predictions = decision_tree.predict(X_test)

# Random Forest
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)

# Task 8: Compare predictions
print("Decision Tree Predictions:", dt_predictions[:10])
print("Random Forest Predictions:", rf_predictions[:10])

# Task 9: Compare accuracy
dt_accuracy = accuracy_score(y_test, dt_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# Task 10: Performance matrix comparison
print("Decision Tree Performance:\n", classification_report(y_test, dt_predictions))
print("Random Forest Performance:\n", classification_report(y_test, rf_predictions))

# Confusion matrices
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
