import pickle

import faiss
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

index = faiss.read_index("./data/processed/index.faiss")

vectors = index.index.reconstruct_n(0, index.ntotal)

# PCAを使用してベクトルを二次元に変換します
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)


# target_mapping.pklをロードします
with open("./data/processed/target_mapping.pkl", "rb") as f:
    target_mapping = pickle.load(f)

# 各ベクトルのIDに対応するラベルを取得します
labels = np.array([target_mapping[i] for i in range(index.ntotal)])

# ラベルが0と1のデータを色分けしてプロットします
plt.scatter(
    vectors_2d[labels == 0, 0], vectors_2d[labels == 0, 1], c="tab:orange", label="0"
)
plt.scatter(
    vectors_2d[labels == 1, 0], vectors_2d[labels == 1, 1], c="tab:cyan", label="1"
)

# 判例を追加します
plt.legend()

plt.show()
