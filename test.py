from milvus import Milvus, DataType
from pprint import pprint
import random
import numpy as np
from bert_serving.client import BertClient
# ip address of the GPU machine
bc = BertClient(ip='10.13.5.221')
s_encode = bc.encode(['写代码不香吗'])

a = s_encode[0].tolist()

# print(a)

_HOST = 'any.moono.vip'
_PORT = '19530'
client = Milvus(_HOST, _PORT)


collection_name = 'demo_films'
if collection_name in client.list_collections():
    client.drop_collection(collection_name)


collection_param = {
    "fields": [
        #  Milvus doesn't support string type now, but we are considering supporting it soon.
        # {"name": "title", "type": DataType.STRING},
        {"name": "duration", "type": DataType.INT32, "params": {"unit": "minute"}},
        {"name": "release_year", "type": DataType.INT32},
        {"name": "myid", "type": DataType.INT32},
        {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 768}},
    ],
    "segment_row_limit": 4096,
    "auto_id": True
}

client.create_collection(collection_name, collection_param)
client.create_partition(collection_name, "American")


# The_Lord_of_the_Rings = [
#     {
#         "title": "The_Fellowship_of_the_Ring",
#         "id": 1,
#         "duration": 208,
#         "release_year": 2001,
#         "embedding": a
#     },
#     {
#         "title": "The_Two_Towers",
#         "id": 2,
#         "duration": 226,
#         "release_year": 2002,
#         "embedding": [random.random() for _ in range(768)]
#     },
#     {
#         "title": "The_Return_of_the_King",
#         "id": 3,
#         "duration": 252,
#         "release_year": 2003,
#         "embedding": [random.random() for _ in range(768)]
#     }
# ]


# ids = [k.get("id") for k in The_Lord_of_the_Rings]
# durations = [k.get("duration") for k in The_Lord_of_the_Rings]
# release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
# embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

# hybrid_entities = [
#     # Milvus doesn't support string type yet, so we cannot insert "title".
#     {"name": "duration", "values": durations, "type": DataType.INT32},
#     {"name": "release_year", "values": release_years, "type": DataType.INT32},
#     {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
# ]


# ids = client.insert(collection_name, hybrid_entities,
#                     ids, partition_tag="American")


# # ------
# # Basic insert entities:
# #     After insert entities into collection, we need to flush collection to make sure its on disk,
# #     so that we are able to retrieve it.
# # ------
# before_flush_counts = client.count_entities(collection_name)
# client.flush([collection_name])
# after_flush_counts = client.count_entities(collection_name)


# query_embedding = [random.random() for _ in range(768)]
# query_hybrid = {
#     "bool": {
#         "must": [
#             # {
#             #     "term": {"release_year": []}
#             # },
#             # {
#             #     # "GT" for greater than
#             #     "range": {"duration": {"GT": 1, "LT": 290}}
#             # },
#             {
#                 "vector": {
#                     "embedding": {"topk": 3, "query": [a], "metric_type": "L2"}
#                 }
#             }
#         ]
#     }
# }
# results = client.search(collection_name, query_hybrid, fields=[
#                         "duration", "release_year", "embedding"])
# print("\n----------search----------")
# for entities in results:
#     for topk_film in entities:
#         current_entity = topk_film.entity
#         print("- id: {}".format(topk_film.id))
#         print("- distance: {}".format(topk_film.distance))

#         print("- release_year: {}".format(current_entity.release_year))
#         print("- duration: {}".format(current_entity.duration))
#         # print("- embedding: {}".format(current_entity.embedding))
