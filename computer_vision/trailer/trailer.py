from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler


df['synopsis'] = 
print("Downloading pre-trained model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Encoding sentences...")
sentences = model.encode(df["overview"].values)
print("Done.")
scaler = StandardScaler()
X = scaler.fit_transform(sentences)


