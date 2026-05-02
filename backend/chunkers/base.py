from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    text: str
    index: int
    parent_section: Optional[str] = None
    chunk_type: str = "text"
    metadata: dict = field(default_factory=dict)
