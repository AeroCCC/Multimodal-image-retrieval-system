import chromadb

# 1. 关键修改：使用 PersistentClient 并指定存储路径
# 这样数据就会保存在当前目录下的 "my_chroma_db" 文件夹里
client = chromadb.PersistentClient(path="./my_chroma_db")

# 2. 获取或创建集合
# get_or_create: 如果有了就直接读取，没有就新建
collection = client.get_or_create_collection(name="test_collection")

# 3. 只有当集合是空的时候，我们才插入数据（避免重复插入）
if collection.count() == 0:
    print("📥 正在写入数据...")
    collection.add(
        documents=["这是关于AI的文章", "这是关于做饭的食谱", "YOLO是目标检测算法"],
        metadatas=[{"type": "tech"}, {"type": "life"}, {"type": "tech"}], # 加上标签
        ids=["doc1", "doc2", "doc3"]
    )
else:
    print("✅ 数据已存在，直接加载！")

# 4. 尝试查询
results = collection.query(
    query_texts=["计算机技术"],
    n_results=2,
    where={"type":"tech"}
)

print(results['documents'])