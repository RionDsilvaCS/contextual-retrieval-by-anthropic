from src.contextual_retrieval import create_and_save_db

data_dir = "/home/rion/agents/contextual-retrieval-by-anthropic/data"
save_dir = "/home/rion/agents/contextual-retrieval-by-anthropic/src/db"
collection_name = "cook_book"
db_name = "cook_book_db"

create_and_save_db(
    data_dir=data_dir, 
    save_dir=save_dir,
    collection_name=collection_name,
    db_name=db_name
    )