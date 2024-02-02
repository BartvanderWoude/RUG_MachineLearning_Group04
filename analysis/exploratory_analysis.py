import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read the training CSV file into a DataFrame
train_df = pd.read_csv('../CBIS-DDSM/csv/calc_case_description_train_set.csv')

# Encode categorical columns ('mass shape' and 'mass margins')
label_encoder = LabelEncoder()
train_df['calc type'] = label_encoder.fit_transform(train_df['calc type'])
train_df['calc distribution'] = label_encoder.fit_transform(train_df['calc distribution'])

# Define features (X_train) and target variable (y_train)
X_train = train_df[['calc type', 'calc distribution','subtlety']]
y_train = (train_df['assessment'] > 3).astype(int)  # 1 if assessment > 3, 0 otherwise

# Read the testing CSV file into a DataFrame
test_df = pd.read_csv('../CBIS-DDSM/csv/calc_case_description_test_set.csv')

# Encode categorical columns in the testing data
test_df['calc type'] = label_encoder.fit_transform(test_df['calc type'])
test_df['calc distribution'] = label_encoder.fit_transform(test_df['calc distribution'])
# test_df['subtlety'] = label_encoder.fit_transform(test_df['subtlety'])

# Define features (X_test) and target variable (y_test)
X_test = test_df[['calc type', 'calc distribution','subtlety']]
y_test = (test_df['assessment'] > 3).astype(int)  # 1 if assessment > 3, 0 otherwise

# Build a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report_str}')
