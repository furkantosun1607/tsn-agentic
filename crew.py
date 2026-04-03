"""
TSN Media AI Crew — CrewAI orchestration module.
Uses modern decorator-based API: @CrewBase, @agent, @task, @crew.
Configured to use Google Gemini 2.5 Flash-Lite as the LLM.
"""

import os
import sys

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Ensure backend is importable
_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from ai_workers.db_tools import (
    GetAvailableCategoriesTool,
    GetPendingNewsTool,
    SaveQualityScoreTool,
    UpdateScoreAndCategoriesTool,
    UpdateSummaryTool,
)
from ai_workers.schemas import CategorizationOutput, ScoreOutput, SummaryOutput

# ---------------------------------------------------------------------------
# LLM Configuration — Google Gemini 2.5 Flash-Lite (override with GEMINI_MODEL)
# ---------------------------------------------------------------------------
gemini_llm = LLM(
    model=os.environ.get("GEMINI_MODEL", "gemini/gemini-2.5-flash-lite"),
    api_key=os.environ.get("GEMINI_API_KEY"),
)


@CrewBase
class TsnMediaCrew:
    """
    TSN Media AI Crew.
    Processes pending news articles through three sequential stages:
    1. Quality Scoring
    2. Categorization (max 3 categories)
    3. Summarization (TL;DR)
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @agent
    def scoring_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["scoring_agent"],
            tools=[SaveQualityScoreTool()],
            llm=gemini_llm,
        )

    @agent
    def categorization_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["categorization_agent"],
            tools=[GetAvailableCategoriesTool()],
            llm=gemini_llm,
        )

    @agent
    def summarization_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["summarization_agent"],
            tools=[],
            llm=gemini_llm,
        )

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------

    @task
    def score_task(self) -> Task:
        return Task(
            config=self.tasks_config["score_task"],
            tools=[SaveQualityScoreTool()],
            output_pydantic=ScoreOutput,
        )

    @task
    def categorize_task(self) -> Task:
        return Task(
            config=self.tasks_config["categorize_task"],
            tools=[
                GetAvailableCategoriesTool(),
                UpdateScoreAndCategoriesTool(),
            ],
            output_pydantic=CategorizationOutput,
        )

    @task
    def summarize_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_task"],
            tools=[UpdateSummaryTool()],
            output_pydantic=SummaryOutput,
        )

    # ------------------------------------------------------------------
    # Crew Assembly
    # ------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        """Assembles the TSN Media AI Crew with sequential processing."""
        return Crew(
            agents=[self.scoring_agent()],  # Sadece puanlama calisir
            tasks=[self.score_task()],      # Kategorize ve summary simdilik calistirilmaz
            process=Process.sequential,
            verbose=True,
        )
