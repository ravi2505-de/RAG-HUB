from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone(api_key, environment, index_name="rag-index", dimension=512):
    pc = Pinecone(api_key=api_key)

    # Check if index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=environment  # e.g., 'us-east-1'
            )
        )
        print(f"✅ Created Pinecone index: {index_name}")
    else:
        print(f"✅ Using existing Pinecone index: {index_name}")
