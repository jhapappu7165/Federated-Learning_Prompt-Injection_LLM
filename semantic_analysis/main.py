import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Load dataset
df = pd.read_csv('datasets/combined_benign_malicious.csv')

# 2. Preprocessing
def preprocess(text):
    text = str(text).lower().strip()
    # Add more preprocessing as needed (remove special chars, etc.)
    return text

df['prompt'] = df['prompt'].apply(preprocess)

# 3. Encode labels
label_map = {'benign': 0, 'malicious': 1}
df['label'] = df['label'].map(label_map)

# 4. Semantic Embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['prompt'].tolist(), show_progress_bar=True)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 6. Train Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 7. Evaluation
y_pred = clf.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['benign', 'malicious']))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# 8. (Optional) Visualization
pca = PCA(n_components=2)
X_vis = pca.fit_transform(embeddings)
plt.figure(figsize=(8,6))
plt.scatter(X_vis[df['label']==0,0], X_vis[df['label']==0,1], label='Benign', alpha=0.5)
plt.scatter(X_vis[df['label']==1,0], X_vis[df['label']==1,1], label='Malicious', alpha=0.5)
plt.legend()
plt.title('PCA of Prompt Embeddings')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show() 