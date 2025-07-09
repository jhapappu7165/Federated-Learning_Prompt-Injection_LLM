from preprocessing import embeddings, df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump


x_train, x_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=42, stratify=df['label'])

# train classification model
model = LogisticRegression(max_iter=1000)

# Model can iterate up to 1000 times to find the best fit (minimized loss function by updating weights' values to reduce this loss)

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

print('***Classification Report***')
print(classification_report(y_test, y_predicted, target_names=['benign', 'malicious']))

print('***Confusion Matrix***')
print(confusion_matrix(y_test, y_predicted))

# Save the trained model
dump(model, 'semantic_analysis/trained_model.joblib')
print('Model saved to semantic_analysis/trained_model.joblib')
