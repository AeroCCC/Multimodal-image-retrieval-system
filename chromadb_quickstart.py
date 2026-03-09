import chromadb

# 1. 创建一个本地客户端 (就像连接 SQLite)
client = chromadb.Client()

# 2. 创建一个集合 (就像数据库里的一张表)
collection = client.create_collection(name="my_knowledge_base")

# 3. 存入数据 (Chroma 会自动把 texts 转换成向量存起来)
collection.add(
    documents=[
        "This is a document about cats and dogs.",  
        "I love eating apples and bananas.",        
        "My car is a red Ferrari.",
        "Where is your car.",
        "I want to konw about animals"                 
    ],
    ids=["id1", "id2", "id3","id4","id5"]
)

# 4. 搜索 (Query)
results = collection.query(
    query_texts=["I want a monkey"],
    n_results=3
)

# 5. 输出结果
print(results['documents'])