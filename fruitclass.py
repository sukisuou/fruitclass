# This cell imports the necessary libraries for data manipulation, visualization, and machine learning.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

# This cell downloads the 'fruit-classification' dataset from KaggleHub and identifies the path to the CSV file within the downloaded dataset.
path = kagglehub.dataset_download("pranavkapratwar/fruit-classification")
print("Path to dataset files:", path)

for file in os.listdir(path):
    if file.endswith(".csv"):
        dataset_path = os.path.join(path, file)
        break
print("Using dataset file:", dataset_path)

# This cell reads the identified CSV file into a pandas DataFrame and then displays the first 5 rows and a summary of its structure (data types, non-null counts).
df = pd.read_csv(dataset_path)
print("Dataset loaded successfully.")
print(df.head())
print(df.info())

# This cell performs label encoding on the specified categorical columns ('shape', 'color', 'taste') to convert them into numerical representations, which is required for many machine learning algorithms.

from sklearn.preprocessing import LabelEncoder, StandardScaler
categorical_cols = ['shape', 'color', 'taste']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# This cell encodes the target variable ('fruit_name') into numerical labels using LabelEncoder, making it suitable for classification models, and then displays the updated DataFrame head.

target_le = LabelEncoder()
df['fruit_name'] = target_le.fit_transform(df['fruit_name'])
print(df.head())

# This cell separates the dataset into features (X) by dropping the 'fruit_name' column and the target variable (Y) containing only 'fruit_name'.
target_column = 'fruit_name'

X = df.drop(target_column, axis=1)
Y = df[target_column]

# This cell splits the feature (X) and target (Y) data into training and testing sets, with 33% of the data reserved for testing, ensuring a consistent split using a random state.
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.33, random_state = 42
)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# This cell initializes a Decision Tree Classifier and trains it using the training data (X_train, Y_train).
from sklearn.tree import DecisionTreeClassifier
treemodel = DecisionTreeClassifier(criterion = 'gini')
treemodel.fit(X_train, Y_train)

# This cell visualizes the trained Decision Tree model, providing a graphical representation of the decision-making process.
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)

# This cell uses the trained Decision Tree model to make predictions on the test set (X_test) and then displays the predicted labels.
y_pred=treemodel.predict(X_test)

y_pred

# This cell calculates and prints the accuracy score by comparing the predicted labels (y_pred) with the actual labels (Y_test), indicating the model's overall performance.
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_pred, Y_test)
print(f"Accuracy score: {score}")

# This cell generates and displays a confusion matrix to evaluate the performance of the classification model, showing the counts of true positive, true negative, false positive, and false negative predictions.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_le.classes_)
disp.plot(cmap=plt.cm.Blues, values_format='d')

# Countplot for fruit_name for distribution

plt.figure(figsize=(12, 6))
# Use the original DataFrame 'df' which still has the encoded 'fruit_name'
ax = sns.countplot(x='fruit_name', data=df, palette='viridis')
plt.title('Distribution of Fruit Names')
plt.xlabel('Fruit Name')
plt.ylabel('Count')

# Get the actual string names from the 'target_le' encoder
ax.set_xticklabels(target_le.classes_, rotation=90)

plt.tight_layout()
plt.show()

# Histogram to check for frequency for numeric data (size, weight and avg_price)

numeric_cols = ['size (cm)', 'weight (g)', 'avg_price (₹)']

plt.figure(figsize=(16, 5))
plt.suptitle('Distribution of Numeric Features', fontsize=16)

for i, col in enumerate(numeric_cols):
    plt.subplot(1, 3, i + 1)

    # Create a histogram
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Classification Report (performance summary to show precision)
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred, target_names=target_le.classes_))

# Different way of scoring model (cross validation: k-fold validation)
from sklearn.model_selection import cross_val_score

# Using 5-fold
scores = cross_val_score(treemodel, X, Y, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Average cross-val score: {scores.mean()}")

# Feature Importance Graph (rank influence of features in learning)
importances = treemodel.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.show()

# Create a new, simpler tree(pruned tree, with only 5 max depth)
pruned_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
pruned_tree.fit(X_train, Y_train)


plt.figure(figsize=(20,10))
tree.plot_tree(pruned_tree, filled=True, feature_names=X.columns, class_names=target_le.classes_)
plt.show()

# check its accuracy
pruned_pred = pruned_tree.predict(X_test)
print(f"Pruned Tree Accuracy: {accuracy_score(pruned_pred, Y_test)}")

# save model
import pickle

# save trained model
with open('treemodel.pkl', 'wb') as file:
    pickle.dump(treemodel, file)