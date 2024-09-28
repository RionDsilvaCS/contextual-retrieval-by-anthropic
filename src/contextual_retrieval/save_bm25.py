from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
import os

def save_BM25(nodes: list, 
              save_dir: str = "./", 
              db_name: str = "none") -> None:
    
    print("-:-:-:- BM25 [TF_IDF Database] creating ... -:-:-:-")

    # Initializing BM25
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=12,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    # Path to save BM25
    save_pth = os.path.join(save_dir, db_name)

    # Saving BM25
    bm25_retriever.persist(save_pth)

    print("-:-:-:- BM25 [TF_IDF Database] saved -:-:-:-")