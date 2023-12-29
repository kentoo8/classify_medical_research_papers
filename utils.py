import pickle
from collections import Counter
from typing import List

import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings


def merge_faiss_indexes(index_paths: List[str], output_path: str) -> None:
    """
    指定されたパスのリストに含まれるすべての.faissファイルを読み込み、
    それらを統合して1つの.faissファイルを生成します。

    Args:
        index_paths (List[str]): .faissファイルのパスのリスト。
        output_path (str): 統合された.faissファイルを保存するパス。
    """
    # 最初のインデックスを読み込む
    index = faiss.read_index(index_paths[0])

    # 残りのインデックスを読み込み、統合する
    for index_path in index_paths[1:3]:
        sub_index = faiss.read_index(index_path)
        index.add(sub_index)

    # 統合されたインデックスを保存
    faiss.write_index(index, output_path)


def get_most_common_target(*, ids: List[int], id_target_dict: dict) -> int:
    """
    最も頻繁に出現する目的変数を取得します。

    Args:
        ids (List[int]): IDのリスト。
        id_target_dict (dict): IDと目的変数のマッピング。

    Returns:
        int: 最も頻繁に出現する目的変数。
    """
    return Counter(id_target_dict[id] for id in ids).most_common(1)[0][0]


def knn_predict(
    *,
    texts: List[str],
    index_path: str,
    target_mapping_path: str,
    k: int = 3,
) -> List[int]:
    """
    k近傍法（k-NN）を使用して目的変数（judgement）を予測します。

    Args:
        texts (List[str]): 予測を行うためのテキストのリスト。
        index_path (str): Faissのインデックスが保存されているパス。
        target_mapping_path (str): IDと目的変数（judgement）のマッピングが保存されているパス。
        k (int): 近傍の数。

    Returns:
        List[int]: 予測結果のリスト。
    """
    # テキストを埋め込む
    embeddings_model = OpenAIEmbeddings()
    embedded = embeddings_model.embed_documents(texts)

    # Faissのインデックスを読み込む
    index = faiss.read_index(index_path)

    # IDと目的変数（judgement）のマッピングを読み込む
    with open(target_mapping_path, "rb") as f:
        id_target_dict = pickle.load(f)

    # 各埋め込みに対する最近傍のベクトルを見つける
    _, indices = index.search(np.array(embedded, dtype=np.float32), k)

    # 最近傍のベクトルのIDを使用して、それに対応する目的変数（judgement）を取得し、最も頻繁に出現する目的変数を予測結果とする
    targets = [
        get_most_common_target(ids=ids, id_target_dict=id_target_dict)
        for ids in indices
    ]

    return targets
