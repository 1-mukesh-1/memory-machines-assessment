"""
Part 2 Configuration - Event definitions, keywords, and chunk settings.
"""

from dataclasses import dataclass, field


@dataclass
class EventConfig:
    """Configuration for a single historical event."""
    id: str
    name: str
    primary_keywords: list[str]  # Must match at least one
    secondary_keywords: list[str] = field(default_factory=list)  # Boost relevance


# The 5 target events from the assignment
EVENTS = [
    EventConfig(
        id="election_1860",
        name="Election Night 1860",
        primary_keywords=["election", "1860"],
        secondary_keywords=["november", "votes", "returns", "telegraph", "springfield", "elected", "president-elect"],
    ),
    EventConfig(
        id="fort_sumter",
        name="Fort Sumter Decision",
        primary_keywords=["sumter"],
        secondary_keywords=["fort", "charleston", "anderson", "pickens", "resupply", "bombardment", "april 1861"],
    ),
    EventConfig(
        id="gettysburg_address",
        name="Gettysburg Address",
        primary_keywords=["gettysburg"],
        secondary_keywords=["cemetery", "dedication", "november 1863", "consecrate", "fourscore", "four score"],
    ),
    EventConfig(
        id="second_inaugural",
        name="Second Inaugural Address",
        primary_keywords=["inaugural"],
        secondary_keywords=["march 1865", "second inaugural", "malice", "charity", "bondsman", "swear"],
    ),
    EventConfig(
        id="fords_theatre",
        name="Ford's Theatre Assassination",
        primary_keywords=["ford", "theatre", "theater", "booth", "assassin"],
        secondary_keywords=["shot", "april 14", "april 15", "petersen", "pistol", "wilkes", "death"],
    ),
]


@dataclass
class ChunkConfig:
    """Text chunking configuration."""
    chunk_size_tokens: int = 1500
    overlap_tokens: int = 200
    top_n_chunks: int = 5  # Max chunks to send to LLM per event
    chars_per_token: float = 4.0  # Rough estimate
    
    @property
    def chunk_size_chars(self) -> int:
        return int(self.chunk_size_tokens * self.chars_per_token)
    
    @property
    def overlap_chars(self) -> int:
        return int(self.overlap_tokens * self.chars_per_token)


@dataclass
class Part2Config:
    """Complete Part 2 configuration."""
    events: list[EventConfig] = field(default_factory=lambda: EVENTS)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    
    # Paths
    input_gutenberg: str = "data/normalized/gutenberg.json"
    input_loc: str = "data/normalized/loc.json"
    output_path: str = "data/extractions"
    
    # LLM settings (overridden by env vars)
    default_provider: str = "anthropic"
    default_model: str = "claude-sonnet-4-20250514"
    prompt_strategy: str = "cot"  # zero_shot | cot | few_shot