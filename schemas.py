"""
Pydantic output models for CrewAI agent responses.
These models enforce structured output from each agent via `output_pydantic`.
"""

from pydantic import BaseModel, Field


class ScoreOutput(BaseModel):
    """Scoring agent's structured output."""
    article_id: int = Field(..., description="İlgili makalenin ID'si")
    score: int = Field(
        ..., ge=1, le=100,
        description="1-100 arası kalite puanı"
    )


class CategorizationOutput(BaseModel):
    """Categorization agent's structured output."""
    article_id: int = Field(..., description="İlgili makalenin ID'si")
    categories: list[str] = Field(
        ..., max_length=3,
        description="En fazla 3 kategori adı listesi"
    )


class SummaryOutput(BaseModel):
    """Summarization agent's structured output."""
    article_id: int = Field(..., description="İlgili makalenin ID'si")
    summary: str = Field(
        ..., description="3-4 maddelik TL;DR özet"
    )
