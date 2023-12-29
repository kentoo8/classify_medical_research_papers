import pickle
import time
from typing import List

import faiss
import numpy as np
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings


class OpenAIEmbeddingsWithRateLimit(OpenAIEmbeddings):
    def embed_documents(
        self, documents: List[str], chunk_size: int = 100, sleep_time: int = 30
    ) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(documents), chunk_size):
            print(i)
            chunk = documents[i : i + chunk_size]
            while True:
                try:
                    embeddings_chunk = super().embed_documents(chunk)
                    embeddings.extend(embeddings_chunk)
                    break
                except Exception as e:  # FIXME: API制限エラーの具体的な例外クラスに置き換えてください
                    print(f"API制限エラーが発生しました: {e}.\n\n{sleep_time}秒間待機します。")
                    time.sleep(sleep_time)
        return embeddings


def get_combined_texts(
    *,
    df: pd.DataFrame,
    text_columns: List[str],
) -> List[str]:
    combined_texts = df[text_columns].apply(
        lambda row: " ".join(row.values.astype(str)), axis=1
    )
    return list(combined_texts)


def save_embeddings(
    *,
    df: pd.DataFrame,
    text_columns: List[str],
    index_path: str,
    id_column: str = "id",
) -> None:
    """
    指定された列を結合してから埋め込み、その結果とIDを紐付けたFaissファイル（バイナリ）として保存します。

    Args:
        df (pd.DataFrame): 埋め込みを行うためのデータが含まれたDataFrame。
        id_column (str): IDが含まれる列の名前。
        text_columns (List[str]): 埋め込み対象のテキストデータが含まれる列の名前のリスト。
        index_path (str): Faissのインデックスを保存するパス。
    """
    combined_texts = get_combined_texts(df=df, text_columns=text_columns)

    # テキストを埋め込む
    embeddings_model = OpenAIEmbeddingsWithRateLimit()
    embedded = embeddings_model.embed_documents(combined_texts)

    # 埋め込みベクトルの次元数
    dimension = len(embedded[0])

    # Faissのインデックスを作成
    index = faiss.IndexFlatL2(dimension)

    # IDとベクトルをNumPy配列に変換
    ids = np.array(df[id_column].values, dtype=np.int64)
    vectors = np.array(embedded, dtype=np.float32)

    # IDMapを作成し、ベクトルとIDを追加
    index = faiss.IndexIDMap(index)
    index.add_with_ids(vectors, ids)

    faiss.write_index(index, index_path)


def save_id_target_dict(
    *,
    df: pd.DataFrame,
    output_path: str,
    id_column: str = "id",
    target_column: str = "target",
) -> None:
    """
    指定されたDataFrameからIDと目的変数の辞書を作成し、
    それを指定されたパスにpickle形式で保存します。

    Args:
        df (pd.DataFrame): IDと目的変数が含まれるDataFrame。
        output_path (str): 保存先のパス。
        id_column (str): IDが含まれる列の名前。
        target_column (str): 目的変数が含まれる列の名前。
    """
    id_target_dict = pd.Series(df[target_column].values, index=df[id_column]).to_dict()
    with open(output_path, "wb") as f:
        pickle.dump(id_target_dict, f)


def main():
    train_data = pd.read_csv("./data/raw/train.csv")

    save_id_target_dict(
        df=train_data,
        output_path="./data/processed/id_target_dict.pkl",
        id_column="id",
        target_column="judgement",
    )

    save_embeddings(
        df=train_data,
        text_columns=["title", "abstract"],
        index_path="./data/processed/train_index.faiss",
        id_column="id",
    )


if __name__ == "__main__":
    main()
