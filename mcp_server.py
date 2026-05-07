"""
TSN MEDIA MCP SERVER (MODEL CONTEXT PROTOCOL)
=============================================
SUNUM NOTU (Hoca İçin Mimari Açıklama):
Bu dosya, sunumda (demo_simulation.py) "MCP Browser" olarak gördüğümüz 
canlı Chrome etkileşiminin ve otonom tarayıcı kontrolünün "Gerçek (Production)"
karşılığıdır.

Neden MCP (Model Context Protocol) Kullanıyoruz?
------------------------------------------------
LLM'ler (Geniş Dil Modelleri) doğaları gereği kapalı sistemlerdir ve 
dış dünyayla etkileşim kuramazlar. MCP, yapay zekaya bir "tarayıcı eli" verir.

Bu dosyada gördüğünüz `read_news_url_visually` aracı, yapay zekanın (Ollama veya CrewAI) 
sanki bir insanmış gibi hedef haber sitesine gidip, sayfayı aşağı yukarı kaydırarak (scroll) 
haberi okumasını ve DOM içerisindeki metni söküp almasını sağlar.

Sunumda terminalde yazan "Initiating Headless Browser Context..." aşaması,
üretim ortamında tam olarak buradaki Playwright asenkron fonksiyonu ile çalışmaktadır.
"""
import sys
import os

# Add the backend directory to sys.path so we can import app modules
_backend_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"
)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

from mcp.server.fastmcp import FastMCP
from ai_workers.db_tools import (
    GetPendingNewsTool,
    GetAvailableCategoriesTool,
    UpdateSummaryTool,
    UpdateScoreAndCategoriesTool
)

# Initialize FastMCP Server
mcp = FastMCP("TsnMedia_MCP_Server")

@mcp.tool()
def get_pending_news(limit: int = 10) -> str:
    """Veritabanından henüz AI tarafından işlenmemiş haberleri getirir."""
    tool = GetPendingNewsTool()
    return tool._run(limit=limit)

@mcp.tool()
def get_available_categories() -> str:
    """Veritabanındaki tüm mevcut kategori adlarını getirir."""
    tool = GetAvailableCategoriesTool()
    return tool._run()

@mcp.tool()
def update_summary(article_id: int, summary: str) -> str:
    """Bir haberin summary alanını AI tarafından üretilen TL;DR özet ile günceller."""
    tool = UpdateSummaryTool()
    return tool._run(article_id=article_id, summary=summary)

@mcp.tool()
def update_score_and_categories(article_id: int, score: int, category_names: list[str]) -> str:
    """Bir haberin AI kalite puanını günceller ve kategorilerini kaydeder."""
    tool = UpdateScoreAndCategoriesTool()
    return tool._run(article_id=article_id, score=score, category_names=category_names)

@mcp.tool()
async def read_news_url_visually(url: str) -> str:
    """
    SUNUMDAKİ ROLÜ: Yapay zekanın "okuma yeteneği".
    Belirtilen URL'yi Playwright ile canlı, görünür bir tarayıcıda açar.
    Sayfanın aşağı kaydırılması animasyonla sağlanır, böylece haberlerin
    tamamı yüklenir ve insan benzeri bir okuma simüle edilir.
    Daha sonra DOM'daki asıl metin (içerik) sökülüp ajana geri döndürülür.
    """
    import asyncio
    from playwright.async_api import async_playwright
    
    try:
        async with async_playwright() as p:
            # Görsel olarak açılması için headless=False
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            
            # Belirtilen adrese git
            await page.goto(url, wait_until="domcontentloaded")
            
            # Haberi okuyormuş efekti vermek için aşağı kaydırma
            for _ in range(3):
                await page.evaluate("window.scrollBy(0, 500)")
                await asyncio.sleep(1.5)
                
            # İçeriği çek
            content = await page.evaluate("document.body.innerText")
            
            # Biraz daha bekleyip kapat
            await asyncio.sleep(1)
            await browser.close()
            
            # İçeriğin tamamı çok uzun olabilir, özet kısmını veya ilk bölümünü alalım
            if content:
                return f"[Tarayıcı Otomasyonu: Sayfa başarıyla okundu!]\n\nİçerik Özeti:\n{content[:800]}...\n\n[DEVAMI VAR]"
            else:
                return "Sayfadan içerik alınamadı."
    except Exception as e:
        return f"Tarayıcı hatası oluştu: {str(e)}"

if __name__ == "__main__":
    # Start the MCP server using standard input/output
    mcp.run()
