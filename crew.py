"""
TSN MEDIA AI CREW — CrewAI ORCHESTRATION MODULE (SUNUM DETAYLARI)
====================================================================
Bu dosya, sunumda (demo_simulation.py) terminalde gosterilen 
'[CrewAI] Task 1: @Categorization_Agent' ve '[LangChain Node] Scoring'
gibi aşamaların *gerçek arka plan* kodlarını barındırır. 

SUNUM İLE BAĞLANTISI (MİMARİ AÇIKLAMA):
---------------------------------------
1. LOKAL LLM (OLLAMA): Sunumda gördüğümüz "Connecting to Local Ollama instance..." 
   kısmı tam olarak bu dosyadaki `_use_local_model` ayarı sayesinde gerçekleşir.
   Tamamen ücretsiz, internet bağlantısı gerektirmeyen ve gizlilik odaklı bir 
   yapay zeka modeli (ör. llama3.2:1b) kullanılarak maliyet sıfıra indirilmiştir.

2. SCORING AGENT (Kalite Puanlaması): Sunumda "Score < 50" ise haberin reddedildiği 
   adım, bu dosyadaki `scoring_agent` tarafından gerçekleştirilir. LangGraph 
   tarafından çağrılır ve haberin kalitesini değerlendirip veritabanına kaydeder.

3. CATEGORIZATION AGENT: Sunumda "Extracting taxonomies..." şeklinde gördüğümüz 
   ajan budur. Haberi okuyup en uygun 3 kategoriyi belirler.

4. SUMMARIZATION AGENT: Haberin uzun metnini kısa madde işaretlerine (bullet points)
   çeviren 'TL;DR' özet ajanıdır. 

Maliyet Odaklı Tasarım:
    - FREE_TIER_MODE=true  -> Yerel (Local) Ollama Modelleri kullanılır (Sıfır maliyet).
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
# LLM Configuration
# ---------------------------------------------------------------------------
# Varsayılan olarak yerel (local) model kullanımını aktif ediyoruz (Ollama).
# Bu sayede ücretsiz, sınırsız ve gizli bir şekilde modelleri makinenizde çalıştırabilirsiniz.
_use_local_model = os.environ.get("USE_LOCAL_MODEL", "true").lower() == "true"
_local_model_name = os.environ.get("LOCAL_MODEL_NAME", "llama3.2:1b") # phi3.5:latest vb. kullanılabilir

if _use_local_model:
    # LangChain ChatOllama kullanarak Ollama'ya bağlanma yaklaşımı:
    # NOT: Bunun için `pip install langchain-ollama` gerekir.
    # from langchain_ollama import ChatOllama
    # default_llm = ChatOllama(model=_local_model_name, base_url="http://localhost:11434")
    
    # CrewAI'nin kendi LLM sınıfı LiteLLM üzerinden Ollama'yı zaten desteklediği için,
    # ekstra kütüphane kurmadan doğrudan bu şekilde de kullanabilirsiniz (önerilen):
    default_llm = LLM(
        model=f"ollama/{_local_model_name}",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    )
else:
    # Eğer USE_LOCAL_MODEL=false yapılırsa eski Gemini yapısına geri döner
    _free_tier_mode = os.environ.get("FREE_TIER_MODE", "true").lower() == "true"
    _default_model = (
        "gemini/gemini-2.0-flash-lite"
        if _free_tier_mode
        else "gemini/gemini-2.5-flash-lite"
    )

    default_llm = LLM(
        model=os.environ.get("GEMINI_MODEL", _default_model),
        api_key=os.environ.get("GEMINI_API_KEY", "dummy-key-if-not-provided"),
        temperature=float(os.environ.get("GEMINI_TEMPERATURE", "0.2")),
    )


@CrewBase
class TsnMediaCrew:
    """
    TSN Media AI Crew.
    Processes pending news articles through three sequential stages:
    1. Quality Scoring
    2. Categorization (max 3 categories)
    3. Summarization (TL;DR)

    demo_simulation.py ile görev eşleşmesi:
        - scoring_agent
          "[LangChain - Ollama] AI Quality Score: 74/100" satırını üretir.
          Production'da SaveQualityScoreTool ile puanı DB'ye yazar.

        - categorization_agent
          "[CrewAI] Task 1: @Categorization_Agent" satırının gerçek ajanıdır.
          Sadece LangGraph route_by_score score >= threshold kararından sonra
          çalıştırılır.

        - summarization_agent
          "[CrewAI] Task 2: @Summarization_Agent" satırının gerçek ajanıdır.
          TL;DR metni üretir ve UpdateSummaryTool ile DB'ye kaydeder.

    Not:
        crew() metodu eski tek parça sıralı kullanım için korunur.
        demo akışıyla birebir uyumlu kullanım LangGraph'ın çağırdığı
        scoring_crew() ve categorize_summary_crew() metodlarıdır.
    """

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    @agent
    def scoring_agent(self) -> Agent:
        """
        SUNUMDAKİ ROLÜ: Haberi okuyup 1 ile 100 arasında bir kalite puanı verir.
        Eğer haber "tık tuzağı (clickbait)", çok kısa veya güvenilmez ise
        düşük puan verir. Sunumda bu adım "AI Quality Score: 74/100" şeklinde görünür.
        """
        return Agent(
            config=self.agents_config["scoring_agent"],
            tools=[SaveQualityScoreTool()],
            llm=default_llm,
        )

    @agent
    def categorization_agent(self) -> Agent:
        """
        SUNUMDAKİ ROLÜ: Kalite eşiğini geçen (Score >= 50) haberlerin hangi 
        kategoriye (Gündem, Spor, Teknoloji vb.) ait olduğunu belirler.
        """
        return Agent(
            config=self.agents_config["categorization_agent"],
            tools=[GetAvailableCategoriesTool()],
            llm=default_llm,
        )

    @agent
    def summarization_agent(self) -> Agent:
        """
        SUNUMDAKİ ROLÜ: Uzun ve okuması zor haber metinlerini 3-4 maddelik
        kısa, net (TL;DR) özetlere dönüştürür.
        """
        return Agent(
            config=self.agents_config["summarization_agent"],
            tools=[],
            llm=default_llm,
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
        """
        Eski tek parça CrewAI akışı.

        Bu metot skor, kategori ve özeti tek sequential Crew içinde çalıştırır.
        LangGraph öncesi mimariyi korumak için bırakılmıştır. Demo akışında
        düşük puanlı haberler kategori/özet adımlarına girmediği için yeni
        production giriş noktası bu metodu değil, aşağıdaki ayrık crew'ları
        kullanır.
        """
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

        Bu ayrım demo_simulation.py içindeki threshold kararını mümkün kılar:
        puan yazıldıktan sonra LangGraph route_by_score karar verir ve düşük
        puanlı haberleri diğer ajanlara göndermeden durdurur.
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
                f"UpdateScoreAndCategoriesTool'u cagirirken score={quality_score} kullan.\n"
                "4. Kategori kararini DB'ye yazmak icin update_score_and_categories aracini MUTLAKA cagir."
            ),
            expected_output=(
                "update_score_and_categories cagrisi yapildi; makale ID ve "
                "secilen kategori adlari DB'ye kaydedildi."
            ),
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
                "5. Ozeti update_summary araci ile MUTLAKA kaydet."
            ),
            expected_output=(
                "update_summary cagrisi yapildi; makale ID ve 3-4 maddelik "
                "kisa TL;DR ozeti DB'ye kaydedildi."
            ),
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
