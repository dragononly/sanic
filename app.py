from sanic import Sanic
from sanic.response import json
from milvus import Milvus, DataType
from pprint import pprint
import random
import numpy as np
from bert_serving.client import BertClient
# ip address of the GPU machine
bc = BertClient(ip='192.168.187.41')
# bc = BertClient(ip='10.13.5.221')
_HOST = 'any.moono.vip'
_PORT = '19530'
client = Milvus(_HOST, _PORT)


app = Sanic("hello_example")


@app.route("/delete")
async def test(request):
    myreq = request.get_args(keep_blank_values=True)
    myid = myreq["myid"][0]
    myid2 = int(myid)
    client.delete_entity_by_id(collection_name='demo_films', ids=[myid2])
    return json({"hello": "ok"})

# 修改新闻删除再插入


@app.route("/change")
async def test(request):
    myreq = request.get_args(keep_blank_values=True)
    str = myreq["string"][0]
    myid = myreq["myid"][0]
    myid2 = int(myid)

    collection_name = 'demo_films'
    client.delete_entity_by_id(collection_name='demo_films', ids=[myid2])

    s_encode = bc.encode([str])
    a = s_encode[0].tolist()
    The_Lord_of_the_Rings = [
        {
            "title": "The_Fellowship_of_the_Ring",
            "id": myid2,
            "release_year": 2001,
            "duration": 209,
            "embedding": a
        },
    ]

    ids = [k.get("id") for k in The_Lord_of_the_Rings]
    duration = [k.get("duration") for k in The_Lord_of_the_Rings]
    release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
    embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

    hybrid_entities = [
        # Milvus doesn't support string type yet, so we cannot insert "title".
        {"name": "duration", "values": duration, "type": DataType.INT32},
        {"name": "release_year", "values": release_years, "type": DataType.INT32},
        {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
    ]

    client.insert(collection_name, hybrid_entities, ids,
                  partition_tag="American")
    return json({"hello": "ok"})

# 插入新闻保存


@app.route("/save")
async def test(request):
    myreq = request.get_args(keep_blank_values=True)
    str = myreq["string"][0]
    myid = myreq["myid"][0]
    myid2 = int(myid)
    print(myid2)
    collection_name = 'demo_films'
    s_encode = bc.encode([str])
    a = s_encode[0].tolist()
    The_Lord_of_the_Rings = [
        {
            "title": "The_Fellowship_of_the_Ring",
            # "id": 1,
            "id": myid2,
            "release_year": 2001,
            "duration": 209,
            "embedding": a
        },
    ]

    ids = [k.get("id") for k in The_Lord_of_the_Rings]
    duration = [k.get("duration") for k in The_Lord_of_the_Rings]
    release_years = [k.get("release_year") for k in The_Lord_of_the_Rings]
    embeddings = [k.get("embedding") for k in The_Lord_of_the_Rings]

    hybrid_entities = [
        # Milvus doesn't support string type yet, so we cannot insert "title".
        {"name": "duration", "values": duration, "type": DataType.INT32},
        {"name": "release_year", "values": release_years, "type": DataType.INT32},
        {"name": "embedding", "values": embeddings, "type": DataType.FLOAT_VECTOR},
    ]

    client.insert(collection_name, hybrid_entities, ids,
                  partition_tag="American")
    return json({"hello": "ok"})


@app.route("/search")
async def test(request):
    myreq = request.get_args(keep_blank_values=True)
    str = myreq["string"][0]
    collection_name = 'demo_films'
    s_encode = bc.encode([str])
    a = s_encode[0].tolist()
    # query_embedding = [random.random() for _ in range(768)]
    query_hybrid = {
        "bool": {
            "must": [
                {
                    "vector": {
                        "embedding": {"topk": 30, "query": [a], "metric_type": "L2"}
                    }
                }
            ]
        }
    }
    results = client.search(collection_name, query_hybrid, fields=[
                            "duration", "release_year", "embedding"])
    print("\n----------search----------")
    x = []
    for entities in results:
        for topk_film in entities:
            current_entity = topk_film.entity
            # print("- id: {}".format(topk_film.id))
            # print("- distance: {}".format(topk_film.distance))
            cab = {"myid": topk_film.id, "distance": topk_film.distance}
            x.append(cab)
            # print("- release_year: {}".format(current_entity.release_year))
            # print("- duration: {}".format(current_entity.myid))
            # print("- embedding: {}".format(current_entity.embedding))

    return json({"hello": x})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, auto_reload=True)
