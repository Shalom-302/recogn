import weaviate
import os

class WeaviateManager:
    def __init__(self):
        self.client = weaviate.connect_to_local(
            host="weaviate",
            port=8080,
            grpc_port=50051
        )
        self._setup_schema()

    def _setup_schema(self):
        if not self.client.collections.exists("Face"):
            self.client.collections.create(
                name="Face",
                # Pas de vectorizer interne car on envoie nos embeddings DeepFace
            )

db = WeaviateManager()