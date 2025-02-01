from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from typing import List
import os

async def pinecone_embedding(texts: List[str], model_name: str = "multilingual-e5-large") -> List[List[float]]:
    """
    Custom embedding function using Pinecone's API.
    :param texts: List of text strings to embed.
    :param model_name: Name of the Pinecone embedding model.
    :return: List of embedding vectors.
    """

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Generate embeddings using Pinecone's inference.embed method
    embeddings = pc.inference.embed(
        model=model_name,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    return [e['values'] for e in embeddings.data]
