import os, html, logging, xml.etree.ElementTree as ET
import gradio as gr
import requests
import psycopg

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("research_app")
DATABASE_URL = os.getenv("DATABASE_URL")
ARXIV_API = "https://export.arxiv.org/api/query"

def connect_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is missing in Render Environment.")
    return psycopg.connect(DATABASE_URL, connect_timeout=10)

def setup_history():
    if not DATABASE_URL:
        return
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_searches (
                    id BIGSERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)

def save_search(query, count):
    if not DATABASE_URL:
        return
    try:
        setup_history()
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO research_searches (query, result_count) VALUES (%s, %s)", (query, count))
    except Exception:
        log.exception("Unable to save search history")

def esc(x):
    return html.escape(str(x or ""), quote=True)

def render_cards(papers):
    if not papers:
        return """<div class="empty"><h3>No papers to display</h3><p>Enter a topic and select Search papers.</p></div>"""
    cards = []
    for n, p in enumerate(papers, 1):
        url = p["link"]
        if not url.startswith(("https://arxiv.org/abs/", "http://arxiv.org/abs/")):
            url = "https://arxiv.org/"
        cards.append(f"""
        <article class="paper">
          <div class="meta"><span>PAPER {n:02d}</span><span>{esc(p['date'])}</span></div>
          <h3>{esc(p['title'])}</h3>
          <p class="authors"><b>Authors</b> · {esc(p['authors'] or 'Not listed')}</p>
          <details><summary>Read abstract</summary><p class="abstract">{esc(p['abstract'] or 'No abstract available.')}</p></details>
          <a class="paper-link" href="{esc(url)}" target="_blank" rel="noopener noreferrer">Read paper on arXiv ↗</a>
        </article>""")
    return '<div class="paper-list">' + ''.join(cards) + '</div>'

def search_papers(query, limit, order):
    query = (query or "").strip()
    if not query:
        return "Please enter a research topic.", render_cards([])
    q = " ".join(query.replace('"', " ").split())
    params = {
        "search_query": f'all:"{q}"',
        "start": 0,
        "max_results": int(limit),
        "sortBy": "submittedDate" if order == "Newest first" else "relevance",
        "sortOrder": "descending",
    }
    try:
        r = requests.get(ARXIV_API, params=params, timeout=30,
                         headers={"User-Agent": "AIResearchScientist/1.0"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        papers = []
        for e in root.findall("a:entry", ns):
            title = " ".join((e.findtext("a:title", "", ns) or "").split())
            abstract = " ".join((e.findtext("a:summary", "", ns) or "").split())
            link = e.findtext("a:id", "", ns) or "https://arxiv.org/"
            date = (e.findtext("a:published", "", ns) or "")[:10]
            authors = [a.findtext("a:name", "", ns) or "" for a in e.findall("a:author", ns)]
            papers.append({"title": title, "abstract": abstract, "link": link,
                           "date": date, "authors": ", ".join(authors[:8]) + (" et al." if len(authors) > 8 else "")})
        save_search(query, len(papers))
        if not papers:
            return "No papers found. Try broader keywords.", render_cards([])
        return f"Found {len(papers)} papers.", render_cards(papers)
    except requests.Timeout:
        return "The paper search timed out. Please try again.", render_cards([])
    except requests.RequestException:
        log.exception("arXiv request failed")
        return "Could not reach arXiv. Please try again shortly.", render_cards([])
    except Exception:
        log.exception("Search failed")
        return "Search failed unexpectedly. Check Render service logs.", render_cards([])

def check_database():
    if not DATABASE_URL:
        return "DATABASE_URL is not configured in Render."
    try:
        setup_history()
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database()")
                name = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'")
                count = cur.fetchone()[0]
        return f"Connected to Neon · Database: {name} · Public tables: {count}"
    except Exception:
        log.exception("Database check failed")
        return "Neon connection failed. Check DATABASE_URL and Render logs."

def show_history():
    if not DATABASE_URL:
        return "DATABASE_URL is not configured."
    try:
        setup_history()
        with connect_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT query, result_count,
                    to_char(searched_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI') || ' UTC'
                    FROM research_searches ORDER BY searched_at DESC LIMIT 20""")
                rows = cur.fetchall()
        if not rows:
            return "No searches saved yet. Search for a topic first."
        return "\n".join(f"{i}. {q} — {n} papers — {when}" for i, (q,n,when) in enumerate(rows,1))
    except Exception:
        log.exception("History load failed")
        return "Could not load history. Check Render service logs."

CSS = """
.gradio-container {max-width:1100px!important;margin:auto!important}
#hero {padding:28px;border-radius:20px;background:linear-gradient(120deg,#101b3c,#315efb);color:white;margin-bottom:18px}
#hero h1 {color:white;font-size:38px;margin:0 0 8px}
#hero p {color:#e7edff;margin:0;font-size:16px}
.paper-list {display:grid;grid-template-columns:1fr;gap:14px}
.paper {background:white;color:#172033;border:1px solid #e1e7f0;border-radius:16px;padding:20px;box-shadow:0 3px 12px #1423460a}
.meta {display:flex;justify-content:space-between;gap:10px;color:#315efb;font-size:11px;font-weight:800;letter-spacing:1px;margin-bottom:10px}
.meta span:last-child {color:#65718a;font-weight:500;letter-spacing:0}
.paper h3 {font-size:20px;line-height:1.4;margin:0 0 10px;color:#172033}
.authors {color:#59657b;font-size:13px;line-height:1.6}
details {border-top:1px solid #edf0f5;padding-top:12px}
summary {cursor:pointer;color:#315efb;font-weight:700}
.abstract {font-size:14px;line-height:1.75;color:#3e4a60}
.paper-link {display:inline-block;margin-top:14px;background:#315efb;color:white!important;padding:10px 14px;border-radius:9px;text-decoration:none!important;font-weight:700;font-size:13px}
.empty {text-align:center;padding:35px;border:1px dashed #ccd5e4;border-radius:16px;color:#65718a}
"""

with gr.Blocks(title="AI Research Scientist", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML('<header id="hero"><h1>AI Research Scientist</h1><p>Discover scientific literature, explore abstracts, and open original papers.</p></header>')
    with gr.Group():
        topic = gr.Textbox(label="Research topic", placeholder="Try: machine learning, computer vision, climate change")
        with gr.Row():
            limit = gr.Dropdown([5,10,20,30], value=10, label="Number of papers")
            order = gr.Radio(["Most relevant","Newest first"], value="Most relevant", label="Sort results")
        search_btn = gr.Button("Search papers", variant="primary")
    status = gr.Markdown("Enter a topic to begin.")
    output = gr.HTML(value=render_cards([]))
    with gr.Accordion("Database and search history", open=False):
        db_btn = gr.Button("Check Neon connection")
        db_status = gr.Textbox(label="Database status", interactive=False)
        hist_btn = gr.Button("Show recent searches")
        hist = gr.Textbox(label="Recent searches", lines=8, interactive=False)
        db_btn.click(check_database, outputs=db_status)
        hist_btn.click(show_history, outputs=hist)
    gr.Markdown("Paper search powered by [arXiv](https://arxiv.org/).")
    search_btn.click(search_papers, inputs=[topic,limit,order], outputs=[status,output])
    topic.submit(search_papers, inputs=[topic,limit,order], outputs=[status,output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT","7860")))
