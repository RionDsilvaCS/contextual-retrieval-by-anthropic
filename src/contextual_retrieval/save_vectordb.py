from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
import chromadb
import os

def save_chromadb(nodes: list, 
                  db_name: str, 
                  collection_name: str = "default", 
                  save_dir: str = "./") -> None:
    
    print("-:-:-:- ChromaDB [Vector Database] creating ... -:-:-:-")

    # Embedding Model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    # Path to save the database file
    save_pth = os.path.join(save_dir, db_name)

    # Initializing Vector Database
    db = chromadb.PersistentClient(path=save_pth)

    # Creating Collection
    chroma_collection = db.get_or_create_collection(collection_name)
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, embed_model=embed_model
    )

    print("-:-:-:- ChromaDB [Vector Database] saved -:-:-:-")
