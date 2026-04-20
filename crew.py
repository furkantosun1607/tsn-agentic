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
    # Crew Assembly — Orijinal (main.py ile uyumlu, DEĞİŞTİRİLMEDİ)
    # ------------------------------------------------------------------

    @crew
    def crew(self) -> Crew:
        """Assembles the TSN Media AI Crew with sequential processing."""
        return Crew(
            agents=[
                self.scoring_agent(),
                self.categorization_agent(),
                self.summarization_agent()
            ],
            tasks=[
                self.score_task(),       # 1. Adım: Önce kalite kontrolü yap
                self.categorize_task(),  # 2. Adım: Puanı aldıysa kategorisini belirle
                self.summarize_task()    # 3. Adım: En son özeti çıkar
            ],
            process=Process.sequential,
            verbose=True,
        )

    # ------------------------------------------------------------------
    # LangGraph İçin Ayrık Crew'lar
    # Bu metotlar YALNIZCA langgraph_flow.py tarafından kullanılır.
    # Mevcut crew() metodu ve main.py tamamen korunur — geriye uyumluluk bozulmaz.
    # ------------------------------------------------------------------

    def scoring_crew(self) -> Crew:
        """
        Sadece puanlama adımını çalıştıran tek görevli crew.
        LangGraph scoring_node tarafından çağrılır.
        SaveQualityScoreTool DB'ye puanı yazar; LangGraph node'u
        ardından DB'den okuyarak state'e ekler.
        """
        return Crew(
            agents=[self.scoring_agent()],
            tasks=[self.score_task()],
            process=Process.sequential,
            verbose=True,
        )

    def categorize_summary_crew(self, quality_score: int) -> Crew:
        """
        Kategorizasyon + Özetleme görevlerini ardı ardına çalıştırır.
        LangGraph'ın categorize_and_summarize_node'u tarafından tetiklenir.

        quality_score: Scoring adımında hesaplanan puan.
            UpdateScoreAndCategoriesTool'un çalışabilmesi için task
            açıklamasına dinamik olarak enjekte edilir.
            Bu sayede tasks.yaml DEĞİŞTİRİLMEZ, main.py BOZULMAZ.
        """
        # Kategorize görevi — quality_score task açıklamasına enjekte edildi
        cat_task = Task(
            description=(
                f"Asagidaki haber makalesini oku ve mevcut kategori listesinden\n"
                f"en uygun olanlari sec.\n\n"
                "Makale ID: {article_id}\n"
                "Baslik: {title}\n"
                "Icerik: {content}\n\n"
                "Mevcut Kategoriler: {available_categories}\n\n"
                "KURALLAR:\n"
                "1. KESINLIKLE en fazla 3 kategori sec.\n"
                "2. Sadece yukaridaki listeden kategori secebilirsin.\n"
                "3. Haberin ana temasi ve alt temaları icin en uygun kategorileri belirle.\n\n"
                f"NOT: Bu haberin kalite puani {quality_score}/100 olarak belirlendi.\n"
                f"UpdateScoreAndCategoriesTool'u cagirirken score={quality_score} kullan."
            ),
            expected_output="Makale ID ve secilen kategori adlari (en fazla 3 adet).",
            agent=self.categorization_agent(),
            tools=[GetAvailableCategoriesTool(), UpdateScoreAndCategoriesTool()],
            output_pydantic=CategorizationOutput,
        )

        # Ozetleme gorevi — tasks.yaml'dan birebir alindi, degistirilmedi
        sum_task = Task(
            description=(
                "Asagidaki haber makalesini dikkatle oku ve kisa bir TL;DR ozeti olustur.\n\n"
                "Makale ID: {article_id}\n"
                "Baslik: {title}\n"
                "Icerik: {content}\n\n"
                "KURALLAR:\n"
                "1. Ozet 3-4 madde icermeli (bullet point formatinda).\n"
                "2. Her madde en fazla 1-2 cumle olmali.\n"
                "3. Haberin ana fikrini, onemli detaylari ve sonucunu kapsamali.\n"
                "4. Gereksiz kelimeler ve tekrarlar olmamali.\n"
                "5. Ozeti update_summary araci ile kaydet."
            ),
            expected_output="Makale ID ve 3-4 maddelik kisa TL;DR ozeti.",
            agent=self.summarization_agent(),
            tools=[UpdateSummaryTool()],
            output_pydantic=SummaryOutput,
        )

        return Crew(
            agents=[self.categorization_agent(), self.summarization_agent()],
            tasks=[cat_task, sum_task],
            process=Process.sequential,
            verbose=True,
        )
