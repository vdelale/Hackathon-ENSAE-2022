import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer



def add_NLP_cols(df, n_PCA):
    print("Downloading pre-trained NLP model...")
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print("Encoding sentences...")
    sentences = model.encode(df["overview"].values)
    print("Done.")
    scaler = StandardScaler()
    X = scaler.fit_transform(sentences)
    pca = PCA(n_components=n_PCA)

    out = pca.fit_transform(X)

    for i in range(n_PCA):
        df["PCA_"+str(i)] = pd.Series(out[:, i])
    
    return df