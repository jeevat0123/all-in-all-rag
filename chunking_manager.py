import re
from pydantic import ValidationError
from typing import List, Any, Optional
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator, ValidationError, PositiveInt

class BaseChunkRequest(BaseModel):
    """Common validators for all chunking methods."""
    data: Any
    chunk_size: Optional[PositiveInt] = None
    overlap: int = 0

    @field_validator('data')
    @classmethod
    def ensure_not_empty(cls, v: Any) -> Any:
        """Step 1: Check if data exists and has content."""
        if v is None:
            raise ValueError("Data cannot be None.")
        if hasattr(v, '__len__') and len(v) == 0:
            raise ValueError("Data source is empty; nothing to chunk.")
        return v

    @field_validator('data')
    @classmethod
    def ensure_valid_type(cls, v: Any) -> Any:
        """Step 2: Check if data type is a sequence we can slice."""
        if not isinstance(v, (str, list, tuple)):
            raise TypeError(f"Unsupported data type: {type(v).__name__}. Expected str or list.")
        return v

    @model_validator(mode='after')
    def validate_logic_bounds(self) -> 'BaseChunkRequest':
        """Step 3: Check logic between multiple fields."""
        if self.chunk_size is not None:
            if self.overlap >= self.chunk_size:
                raise ValueError(f"Overlap ({self.overlap}) must be smaller than Chunk Size ({self.chunk_size}).")
        return self

# --- 2. Logic Layer ---

class ChunkingManager:
    def __init__(self, data: Any):
        self.data = data

    def _validate(self, **kwargs) -> Optional[BaseChunkRequest]:
        """Internal helper to trigger the validation pipeline."""
        try:
            return BaseChunkRequest(data=self.data, **kwargs)
        except (ValidationError, ValueError, TypeError) as e:
            logger.error(f"Validation Failed: {e}")
            return None

    def fixed_length_chunking(self, chunk_size: int, overlap: int = 0) -> List[Any]:
        """
        Splits data into fixed-sized blocks with an optional sliding window overlap.

        Args:
            chunk_size (int): Max units (chars/items) per chunk.
            overlap (int): Units to carry over to the next chunk to preserve context.
        """
        valid = self._validate(chunk_size=chunk_size, overlap=overlap)
        if not valid:
            return []

        chunks = []
        data_len = len(valid.data)
        # Calculate the distance the window moves forward
        step = valid.chunk_size - valid.overlap
        
        for i in range(0, data_len, step):
            chunk = valid.data[i : i + valid.chunk_size]
            chunks.append(chunk)
            
            # Stop if the current window covers the remainder of the data
            if i + valid.chunk_size >= data_len:
                break
        
        logger.info(f"Fixed-Length: Created {len(chunks)} chunks (Size: {chunk_size}, Overlap: {overlap}).")
        return chunks

    def paragraph_chunking(self, separator: str = r"\n\n", strip: bool = True) -> List[str]:
        """
        Splits the document at paragraph boundaries using a separator.

        Args:
            separator (str): Regex pattern for paragraph breaks (default double newline).
            strip (bool): Whether to remove leading/trailing whitespace from chunks.
        """
        valid = self._validate()
        if not valid:
            return []
        
        if not isinstance(valid.data, str):
            logger.error("Paragraph chunking requires string input.")
            return []

        # Split using regex to handle multiple newlines robustly
        paragraphs = re.split(separator, valid.data)
        
        # Clean up chunks
        if strip:
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        else:
            paragraphs = [p for p in paragraphs if p]

        logger.info(f"Paragraph-Level: Created {len(paragraphs)} chunks using separator '{separator}'.")
        return paragraphs


# --- 3. Execution ---

if __name__ == "__main__":
    # --- Example 1: Fixed-Length on Text ---
    text_content = "The quick brown fox jumps over the lazy dog."
    mgr_text = ChunkingManager(text_content)
    
    logger.info("Running Fixed-Length Chunking...")
    text_chunks = mgr_text.fixed_length_chunking(chunk_size=15, overlap=5)
    for idx, c in enumerate(text_chunks):
        print(f"  Chunk {idx}: '{c}'")

    print("-" * 30)

    # --- Example 2: Paragraph-Level Chunking ---
    document = """First Paragraph: Introduction to the topic.
    
    Second Paragraph: Detailed explanation of the logic.
    
    This paragraph has multiple lines.


    Third Paragraph: Final conclusion and summary."""
    
    mgr_doc = ChunkingManager(document)
    logger.info("Running Paragraph Chunking...")
    para_chunks = mgr_doc.paragraph_chunking()
    for idx, p in enumerate(para_chunks):
        print(f"  Para {idx+1}: {p}...")
