import pandas as pd
import numpy as np
rng = np.random.default_rng(seed=19530)
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

HOST = 'localhost'
PORT = '19530'
FIELDS = ["id", "speaker", "start", "end", "text", "episode", "embeddings"]

#### Connection
def create_connection():
    print(f"\nCreate connection...")
    connections.connect(host=HOST, port=PORT)
    print(f"\nList connections:")
    print(connections.list_connections())


def create_collections():
    segment_fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=4),
        FieldSchema(name="start", dtype=DataType.DOUBLE),
        FieldSchema(name="end", dtype=DataType.DOUBLE),
        FieldSchema(name="speaker", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="episode", dtype=DataType.INT64)
    ]

    # Collection Schema
    segment_schema = CollectionSchema(fields=segment_fields, enable_dynamic_field=False, auto_id=True, description="segment")

    # Collection
    exist = utility.has_collection("segment")

    # If it exists, drop and recreate
    if exist:
        print('exist')
        collection_segment = Collection("segment")
        collection_segment.drop()
        # Recreate
        collection_segment = Collection("segment", segment_schema)
    else:
        collection_segment = Collection("segment", segment_schema)

    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }

    collection_segment.create_index("embeddings", index)

    return(collection_segment)


#### CRUD Operations

# Get Transcription By FileId
def get_collection_data(collection_name:str, limit=10):
    collection = None
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        print("Collection not available!")

    collection.load()
    result = collection.query(expr="", limit=limit, output_fields=FIELDS)

    return(result)

def get_collection_list():
    return(utility.list_collections())

# Insert Transcription
def insert(collection_name, rows):
    segment = Collection(collection_name)
    segment.insert(rows)

    # After final entity is inserted, it is best to call flush to have no growing segments left in memory
    segment.flush()

    print(f"Number of entities in DB: {segment.num_entities}")


# Update Transcription

# Delete Transcription
def delete_item_by_id(collection_name:str, id):
    collection = None
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        print("Collection not available!")

    result = collection.query(expr=f"id=={id}")
    print(f"query before delete by expr=`{expr}` -> result: \n-{result[0]}\n-{result[1]}\n")

    collection.delete(expr=f"id=={id}")

# Drop Collection
def drop_collection(collection_name:str):
    print(fmt.format("Drop collection " + collection_name))
    utility.drop_collection(collection_name)

def main():
    # create a connection
    create_connection()
    create_collections()


if __name__ == '__main__':
    main()
