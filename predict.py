import pandas as pd

from utils import knn_predict

df = pd.read_csv("./data/raw/test.csv").fillna("").head(15)
df["combined"] = df["title"] + " " + df["abstract"]

texts = list(df["combined"])
predicts = knn_predict(
    texts=texts,
    index_path="./data/processed/index.faiss",
    target_mapping_path="./data/processed/target_mapping.pkl",
    k=3,
)
df["predict"] = predicts

df[["id", "predict"]].to_csv("./data/processed/predict.csv", index=False, header=False)
