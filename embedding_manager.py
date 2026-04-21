from loguru import logger
from functools import cached_property
from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    """Manages HuggingFace embedding models with lazy initialization."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Lazily initializes the model only when first accessed.
        This prevents heavy memory usage during object instantiation.
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            return HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as e:
            logger.exception(f"Failed to initialize model {self.model_name}: {e}")
            raise RuntimeError("Embedding model initialization failed.") from e

    def generate_embeddings(self, contents: list[str]) -> list[list[float]]:
        """Generates embeddings for a list of strings."""
        if not contents:
            return []
            
        logger.info(f"Generating embeddings for {len(contents)} documents")
        return self.model.embed_documents(contents)

# 1. Initialize (The model is NOT loaded into RAM yet)
manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Define your data
docs = ["Hello world", "How to use LangChain with HuggingFace?"]

# 3. Generate (The model loads NOW, only on this first call)
embeddings = manager.generate_embeddings(docs)

logger.info(f"Generated {len(embeddings)} embeddings.")
logger.info(f"Vector size: {len(embeddings[0])}")