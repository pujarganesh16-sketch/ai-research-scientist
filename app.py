"""
AI Research Scientist — Ultimate Research Workspace
Production-minded single-service Gradio application.

Notes:
- External services can still be unavailable; errors are handled rather than hidden.
- Never put credentials in source code. Configure DATABASE_URL and optional OPENAI_API_KEY
  in the hosting provider's environment settings.
"""
from __future__ import annotations

import csv
import html
import io
import json
import logging
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import gradio as gr
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import psycopg
except ImportError:
    psycopg = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

APP_NAME = "AI Research Scientist"
APP_VERSION = "3.0.0"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai_research_scientist")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": f"{APP_NAME}/{APP_VERSION} (academic discovery; contact site administrator)"
})
_RETRY = Retry(
    total=3, connect=3, read=2, status=3, backoff_factor=0.7,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    respect_retry_after_header=True,
)
SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY, pool_connections=10, pool_maxsize=10))

MAX_QUERY_CHARS = 180
MAX_RESULTS = 30
MAX_ABSTRACT_CHARS = 12_000
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself his
how i if in into is it its itself just more most my no nor not of off on once only
or other our ours ourselves out over own same she should so some such than that the
their theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with would
you your
""".split())


@dataclass(frozen=True)
class Paper:
    title: str
    authors: tuple[str, ...]
    abstract: str
    url: str
    published: str
    updated: str
    categories: tuple[str, ...]
    doi: str = ""

    @property
    def year(self) -> int | None:
        try:
            return int(self.published[:4])
        except (ValueError, TypeError):
            return None

    @property
    def author_text(self) -> str:
        if len(self.authors) <= 8:
            return ", ".join(self.authors) or "Not listed"
        return ", ".join(self.authors[:8]) + f" et al. ({len(self.authors)} authors)"


def escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def safe_arxiv_url(value: str) -> str:
    parsed = urlparse(value or "")
    if parsed.scheme in {"https", "http"} and parsed.netloc in {"arxiv.org", "export.arxiv.org"}:
        return value
    return "https://arxiv.org/"


def normalize_query(value: str) -> str:
    value = " ".join((value or "").replace("\x00", " ").split())
    if len(value) > MAX_QUERY_CHARS:
        raise ValueError(f"Please keep the search query under {MAX_QUERY_CHARS} characters.")
    return value


def arxiv_expression(query: str, field: str) -> str:
    # Escape quotes and backslashes so user text cannot accidentally break the API expression.
    q = query.replace("\\", " ").replace('"', " ").strip()
    field_code = {
        "All fields": "all",
        "Title": "ti",
        "Author": "au",
        "Abstract": "abs",
    }.get(field, "all")
    return f'{field_code}:"{q}"' if field_code != "all" else f'all:"{q}"'


def parse_arxiv(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM):
        title = " ".join((entry.findtext("a:title", "", ATOM) or "").split())
        abstract = " ".join((entry.findtext("a:summary", "", ATOM) or "").split())
        url = safe_arxiv_url(entry.findtext("a:id", "", ATOM) or "")
        published = (entry.findtext("a:published", "", ATOM) or "")[:10]
        updated = (entry.findtext("a:updated", "", ATOM) or "")[:10]
        authors = tuple(
            " ".join((author.findtext("a:name", "", ATOM) or "").split())
            for author in entry.findall("a:author", ATOM)
            if (author.findtext("a:name", "", ATOM) or "").strip()
        )
        categories = tuple(
            c.attrib["term"] for c in entry.findall("a:category", ATOM)
            if c.attrib.get("term")
        )
        doi = ""
        for link in entry.findall("a:link", ATOM):
            if link.attrib.get("title", "").lower() == "doi":
                doi = link.attrib.get("href", "")
                break
        if title:
            papers.append(Paper(title, authors, abstract[:MAX_ABSTRACT_CHARS], url,
                                published, updated, categories, doi))
    return papers


def db_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")
    if psycopg is None:
        raise RuntimeError("psycopg is not installed.")
    return psycopg.connect(DATABASE_URL, connect_timeout=8)


def ensure_tables(cur):
    # Additive only: never drops, truncates, or alters existing user tables.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_searches (
            id BIGSERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            field TEXT NOT NULL DEFAULT 'All fields',
            result_count INTEGER NOT NULL DEFAULT 0,
            elapsed_ms INTEGER NOT NULL DEFAULT 0,
            searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_bookmarks (
            id BIGSERIAL PRIMARY KEY,
            arxiv_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            authors TEXT NOT NULL DEFAULT '',
            published TEXT NOT NULL DEFAULT '',
            saved_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


def save_search(query: str, field: str, count: int, elapsed_ms: int) -> None:
    if not DATABASE_URL or psycopg is None:
        return
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                ensure_tables(cur)
                cur.execute(
                    "INSERT INTO research_searches(query, field, result_count, elapsed_ms) VALUES (%s,%s,%s,%s)",
                    (query[:MAX_QUERY_CHARS], field[:40], count, max(0, elapsed_ms)),
                )
    except Exception:
        log.exception("Unable to save search history; search results remain available")


def request_arxiv(params: dict[str, Any]) -> bytes:
    response = SESSION.get(ARXIV_API, params=params, timeout=(8, 40))
    response.raise_for_status()
    if len(response.content) > 12_000_000:
        raise RuntimeError("The arXiv response was unexpectedly large.")
    return response.content


def perform_search(query: str, field: str, sort: str, limit: int):
    start = time.monotonic()
    query = normalize_query(query)
    if not query:
        return [], "Enter a topic to start your research.", 0
    try:
        limit = max(1, min(int(limit), MAX_RESULTS))
    except (ValueError, TypeError):
        limit = 10
    params = {
        "search_query": arxiv_expression(query, field),
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate" if sort == "Newest first" else "relevance",
        "sortOrder": "descending",
    }
    try:
        xml_bytes = request_arxiv(params)
        papers = parse_arxiv(xml_bytes)
        elapsed = int((time.monotonic() - start) * 1000)
        save_search(query, field, len(papers), elapsed)
        if papers:
            status = f"**{len(papers)} papers found** · {elapsed / 1000:.1f}s · Search field: {escape(field)}"
        else:
            status = "No papers found. Try shorter or broader keywords."
        log.info("Search completed: field=%s count=%d elapsed_ms=%d", field, len(papers), elapsed)
        return papers, status, elapsed
    except requests.Timeout:
        log.warning("arXiv timed out")
        return [], "arXiv timed out. Wait a moment and retry; the external service may be busy.", 0
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        log.warning("arXiv HTTP error: %s", code)
        if code == 429:
            message = "arXiv is rate-limiting requests. Please wait briefly before trying again."
        else:
            message = f"arXiv returned an HTTP error ({code}). Please retry later."
        return [], message, 0
    except requests.RequestException:
        log.exception("Network error contacting arXiv")
        return [], "Could not reach arXiv. Check your connection or try again later.", 0
    except ET.ParseError:
        log.exception("Malformed XML received from arXiv")
        return [], "arXiv returned data that could not be parsed. Please try again later.", 0
    except ValueError as exc:
        return [], escape(str(exc)), 0
    except Exception:
        log.exception("Unexpected search failure")
        return [], "An unexpected error occurred. Check service logs if the issue continues.", 0


def render_papers(papers: list[Paper]) -> str:
    if not papers:
        return """
        <section class="empty-state">
          <div class="empty-symbol">⌕</div>
          <h3>Your research workspace is ready</h3>
          <p>Search scholarly literature, inspect abstracts, compare evidence, and export references.</p>
        </section>"""
    cards = []
    for idx, paper in enumerate(papers, 1):
        categories = "".join(f'<span class="tag">{escape(c)}</span>' for c in paper.categories[:5])
        abstract = escape(paper.abstract or "No abstract supplied.")
        url = escape(safe_arxiv_url(paper.url))
        doi = f'<span class="doi">DOI: {escape(paper.doi)}</span>' if paper.doi else ""
        cards.append(f"""
        <article class="paper-card">
          <div class="paper-top"><span class="index">RESEARCH PAPER {idx:02d}</span>
          <span class="date">{escape(paper.published or "Date unavailable")}</span></div>
          <h3>{escape(paper.title)}</h3>
          <p class="authors">{escape(paper.author_text)}</p>
          <div class="tag-row">{categories}</div>
          <details><summary>Explore abstract <span>＋</span></summary><p class="abstract">{abstract}</p></details>
          <div class="paper-bottom"><a href="{url}" target="_blank" rel="noopener noreferrer">Open original paper ↗</a>{doi}</div>
        </article>""")
    return '<div class="papers-grid">' + "".join(cards) + "</div>"


def term_statistics(papers: list[Paper], top_n: int = 12) -> list[tuple[str, int]]:
    tokens = []
    for paper in papers:
        # Use titles + abstracts; this is descriptive frequency, not semantic inference.
        tokens.extend(re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", f"{paper.title} {paper.abstract}".lower()))
    counts = Counter(t for t in tokens if t not in STOPWORDS and not t.isdigit())
    return counts.most_common(top_n)


def entropy_bits(papers: list[Paper]) -> float:
    """Shannon entropy of category frequencies, measured in bits."""
    categories = [c for p in papers for c in p.categories]
    if not categories:
        return 0.0
    counts = Counter(categories)
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def build_analysis(papers: list[Paper], query: str) -> str:
    if not papers:
        return "Run a successful search to generate the research analytics."
    years = [p.year for p in papers if p.year is not None]
    year_range = f"{min(years)}–{max(years)}" if years else "not available"
    authors = [a for p in papers for a in p.authors]
    unique_authors = len(set(authors))
    stats = term_statistics(papers)
    terms = ", ".join(f"`{word}` ({count})" for word, count in stats) or "No frequent terms found."
    cats = Counter(c for p in papers for c in p.categories)
    top_cats = ", ".join(f"`{c}` ({n})" for c, n in cats.most_common(6)) or "Not supplied"
    return (
        f"### Research landscape: {escape(query)}\n\n"
        f"| Metric | Result |\n|---|---:|\n"
        f"| Papers in this result set | {len(papers)} |\n"
        f"| Publication year span | {year_range} |\n"
        f"| Distinct authors in result set | {unique_authors} |\n"
        f"| Category entropy | {entropy_bits(papers):.2f} bits |\n\n"
        f"**Frequent terms**\n\n{terms}\n\n"
        f"**Frequent arXiv categories**\n\n{top_cats}\n\n"
        "> Interpretation note: category entropy measures how broadly category labels are distributed; "
        "word counts describe the returned text. Neither metric proves quality, novelty, causation, or consensus."
    )


def compare_papers(papers: list[Paper], chosen: list[str] | None) -> str:
    chosen = chosen or []
    selected = [p for p in papers if p.url in chosen][:4]
    if len(selected) < 2:
        return "Select at least **two papers** from the current results to compare them."
    rows = []
    for p in selected:
        rows.append(f"| {escape(p.title)} | {escape(p.published or '—')} | {escape(p.author_text)} | {escape(', '.join(p.categories) or '—')} | [Open paper]({safe_arxiv_url(p.url)}) |")
    return "### Side-by-side paper comparison\n\n| Title | Published | Authors | Categories | Source |\n|---|---|---|---|---|\n" + "\n".join(rows)


def bookmarks_add(papers: list[Paper], chosen: list[str] | None) -> str:
    chosen = chosen or []
    selected = [p for p in papers if p.url in chosen]
    if not selected:
        return "Choose one or more papers from the current results first."
    if not DATABASE_URL or psycopg is None:
        return "Bookmarks need Neon: configure DATABASE_URL and install psycopg."
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                ensure_tables(cur)
                for p in selected:
                    cur.execute("""
                        INSERT INTO research_bookmarks(arxiv_url,title,authors,published)
                        VALUES (%s,%s,%s,%s)
                        ON CONFLICT (arxiv_url) DO NOTHING
                    """, (safe_arxiv_url(p.url), p.title, p.author_text, p.published))
        return f"Saved {len(selected)} selected paper(s) to your shared research library."
    except Exception:
        log.exception("Bookmark save failed")
        return "Could not save bookmarks. Check Neon connection and Render logs."


def load_bookmarks() -> str:
    if not DATABASE_URL or psycopg is None:
        return "Configure DATABASE_URL in Render to use the persistent bookmark library."
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                ensure_tables(cur)
                cur.execute("""SELECT title, authors, published, arxiv_url
                               FROM research_bookmarks ORDER BY saved_at DESC LIMIT 100""")
                rows = cur.fetchall()
        if not rows:
            return "Your library is empty. Search and bookmark papers to build it."
        lines = ["### Saved research library"]
        for title, authors, published, url in rows:
            lines.append(f"- **{escape(title)}** ({escape(published or 'date unavailable')}) — {escape(authors)} · [Open]({safe_arxiv_url(url)})")
        return "\n".join(lines)
    except Exception:
        log.exception("Bookmark load failed")
        return "Could not load the saved library. Check database settings and logs."


def export_csv(papers: list[Paper]):
    if not papers:
        raise gr.Error("Search for papers before exporting.")
    path = os.path.join(tempfile.gettempdir(), f"research_export_{int(time.time())}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["title", "authors", "published", "updated", "categories", "abstract", "doi", "url"])
        writer.writeheader()
        for p in papers:
            writer.writerow({
                "title": p.title, "authors": p.author_text, "published": p.published,
                "updated": p.updated, "categories": "; ".join(p.categories),
                "abstract": p.abstract, "doi": p.doi, "url": safe_arxiv_url(p.url)
            })
    return path


def export_bibtex(papers: list[Paper]):
    if not papers:
        raise gr.Error("Search for papers before exporting.")
    path = os.path.join(tempfile.gettempdir(), f"research_refs_{int(time.time())}.bib")
    entries = []
    for i, p in enumerate(papers, 1):
        raw_id = p.url.rstrip("/").split("/")[-1] or str(i)
        key = "arxiv" + re.sub(r"[^A-Za-z0-9:_-]", "", raw_id)
        def b(v: str) -> str:
            return str(v or "").replace("\\", r"\textbackslash{}").replace("{", r"\{").replace("}", r"\}").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        year = p.published[:4] if p.published else ""
        entries.append(f"@misc{{{key},\n  title = {{{b(p.title)}}},\n  author = {{{b(' and '.join(p.authors))}}},\n  year = {{{year}}},\n  howpublished = {{arXiv}},\n  url = {{{safe_arxiv_url(p.url)}}}\n}}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(entries) + "\n")
    return path


def database_status() -> str:
    if not DATABASE_URL:
        return "Neon is not configured (DATABASE_URL missing). Search and exports can still work."
    if psycopg is None:
        return "The PostgreSQL driver is not installed."
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                ensure_tables(cur)
                cur.execute("SELECT current_database(), current_timestamp")
                db_name, now = cur.fetchone()
                cur.execute("""SELECT COUNT(*) FROM information_schema.tables
                               WHERE table_schema='public' AND table_type='BASE TABLE'""")
                count = cur.fetchone()[0]
        return f"**Connected** · Database `{escape(db_name)}` · {count} public tables · server time {escape(now)}"
    except Exception:
        log.exception("Database health check failed")
        return "Neon connection failed. Verify DATABASE_URL in Render environment settings; do not share it publicly."


def history_text() -> str:
    if not DATABASE_URL or psycopg is None:
        return "Configure Neon to view persistent search history."
    try:
        with db_connect() as conn:
            with conn.cursor() as cur:
                ensure_tables(cur)
                cur.execute("""SELECT query, field, result_count, elapsed_ms,
                    to_char(searched_at AT TIME ZONE 'UTC','YYYY-MM-DD HH24:MI') || ' UTC'
                    FROM research_searches ORDER BY searched_at DESC LIMIT 25""")
                rows = cur.fetchall()
        if not rows:
            return "No saved searches yet."
        return "\n".join(f"{i}. **{escape(q)}** · {escape(field)} · {n} results · {ms} ms · {escape(when)}"
                        for i, (q, field, n, ms, when) in enumerate(rows, 1))
    except Exception:
        log.exception("Search history failed")
        return "Could not load history. Verify Neon and check Render logs."


def ai_summary(papers: list[Paper], query: str) -> str:
    if not papers:
        return "Search for papers first."
    if OpenAI is None or not OPENAI_API_KEY:
        return "AI synthesis is optional and not enabled. Add `OPENAI_API_KEY` in Render and include the `openai` package to enable it. Current analysis is deterministic text statistics, not an AI-generated summary."
    # Bound input and result size; only summarize abstracts actually returned by arXiv.
    source = "\n\n".join(f"TITLE: {p.title}\nABSTRACT: {p.abstract[:1800]}\nSOURCE: {safe_arxiv_url(p.url)}" for p in papers[:8])
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=35, max_retries=1)
        result = client.responses.create(
            model=OPENAI_MODEL,
            input=(
                "You are a careful research assistant. Summarize only evidence in the supplied paper titles and abstracts. "
                "Do not invent methods, results, citations, or consensus. Clearly label uncertainty and missing details. "
                "Use sections: Common themes, Differences, Evidence limitations, Follow-up questions. "
                "Cite claims inline using the supplied arXiv URLs. If abstracts do not support a claim, say so.\n\n"
                f"User research topic: {query[:MAX_QUERY_CHARS]}\n\nPapers:\n{source}"
            ),
            max_output_tokens=900,
        )
        text = getattr(result, "output_text", "") or ""
        return text.strip() or "The AI service returned an empty response."
    except Exception:
        log.exception("AI synthesis request failed")
        return "AI synthesis failed. Search results are still available; check the optional AI provider settings and logs."


CSS = r"""
.gradio-container{max-width:1240px!important;margin:auto!important}
#hero{position:relative;overflow:hidden;padding:40px 34px;border-radius:28px;background:radial-gradient(circle at 90% 5%,#6b83ff 0,transparent 34%),linear-gradient(120deg,#10172e,#243a83 62%,#4255c8);color:white;margin:0 0 20px;box-shadow:0 22px 60px #1c2e6428}
#hero .eyebrow{color:#dce5ff;text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;margin:0 0 12px}
#hero h1{color:white;font-size:clamp(31px,5vw,48px);line-height:1.08;margin:0 0 14px;letter-spacing:-1.2px}
#hero p{color:#e6ecff;line-height:1.8;max-width:780px;margin:0}
#hero .hero-meta{display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}
#hero .hero-meta span{font-size:11px;color:white;background:#ffffff18;border:1px solid #ffffff2e;border-radius:999px;padding:7px 10px}
.search-panel{border:1px solid #e1e7f2;border-radius:22px;padding:22px;background:var(--background-fill-primary)}
.papers-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:10px}
.paper-card{min-width:0;padding:22px;border:1px solid #e1e7f0;border-radius:20px;background:var(--background-fill-primary);box-shadow:0 8px 25px #15254b0a;transition:transform .18s ease,box-shadow .18s ease}
.paper-card:hover{transform:translateY(-2px);box-shadow:0 14px 30px #15254b12}
.paper-top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:13px}
.index{font-size:10px;font-weight:900;letter-spacing:1.3px;color:#5969e8}
.date{font-size:11px;color:var(--body-text-color-subdued)}
.paper-card h3{font-size:18px;line-height:1.45;margin:0 0 10px;color:var(--body-text-color)}
.authors{font-size:12px;line-height:1.65;color:var(--body-text-color-subdued);margin-bottom:13px}
.tag-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:15px}
.tag{font-size:10px;border-radius:999px;padding:5px 8px;color:#3b4ca7;background:#eef1ff}
.paper-card details{border-top:1px solid var(--border-color-primary);padding-top:12px}
.paper-card summary{cursor:pointer;font-size:13px;font-weight:750;color:#6572e9;display:flex;justify-content:space-between}
.abstract{font-size:13px;line-height:1.8;margin-top:12px;color:var(--body-text-color)}
.paper-bottom{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap;margin-top:17px}
.paper-bottom a{background:#5367ed;color:#fff!important;text-decoration:none!important;border-radius:10px;padding:10px 13px;font-size:12px;font-weight:800}
.doi{font-size:10px;color:var(--body-text-color-subdued);overflow-wrap:anywhere}
.empty-state{text-align:center;padding:46px 20px;border:1px dashed #cbd5e6;border-radius:22px;background:var(--background-fill-secondary)}
.empty-symbol{font-size:38px;color:#6272f0}.empty-state h3{margin:10px 0;color:var(--body-text-color)}.empty-state p{color:var(--body-text-color-subdued);line-height:1.7}
footer{visibility:hidden}
@media(max-width:800px){.papers-grid{grid-template-columns:1fr}#hero{padding:29px 21px;border-radius:21px}.paper-card{padding:18px}}
"""

with gr.Blocks(title=f"{APP_NAME} | Premium Research Workspace", theme=gr.themes.Soft(), css=CSS) as demo:
    papers_state = gr.State([])
    chosen_state = gr.State([])
    gr.HTML(
        '<header id="hero"><p class="eyebrow">Academic intelligence workspace · v3.0</p>'
        '<h1>AI Research Scientist</h1>'
        '<p>Explore scholarly literature, inspect source abstracts, compare studies, quantify the research landscape, and export citations in a polished workspace.</p>'
        '<div class="hero-meta"><span>arXiv discovery</span><span>Research analytics</span><span>CSV + BibTeX</span><span>Optional Neon library</span></div></header>'
    )
    with gr.Group(elem_classes=["search-panel"]):
        query = gr.Textbox(label="Research topic", placeholder="e.g., robust machine learning for medical imaging", lines=1, max_lines=2)
        with gr.Row():
            field = gr.Dropdown(["All fields", "Title", "Author", "Abstract"], value="All fields", label="Search field")
            limit = gr.Dropdown([5, 10, 20, 30], value=10, label="Maximum results")
            sort = gr.Radio(["Most relevant", "Newest first"], value="Most relevant", label="Order")
        with gr.Row():
            search_button = gr.Button("Explore papers", variant="primary")
            clear_button = gr.Button("Clear workspace")
    status = gr.Markdown("Ready when you are. Enter a topic to discover research.")
    papers_html = gr.HTML(render_papers([]))
    with gr.Accordion("Research analytics", open=True):
        analytics = gr.Markdown("A quantitative overview appears after a successful search.")
    with gr.Accordion("Compare papers", open=False):
        gr.Markdown("Select two or more papers from the current search results. Comparison uses available metadata.")
        paper_choices = gr.CheckboxGroup(choices=[], label="Papers to compare or save", interactive=True)
        compare_button = gr.Button("Compare selected papers")
        comparison = gr.Markdown("Select at least two papers.")
        save_button = gr.Button("Save selected papers to Neon library")
        bookmark_status = gr.Markdown("")
    with gr.Accordion("AI-assisted synthesis (optional)", open=False):
        gr.Markdown("Optional AI summaries require an API key and provider usage may incur charges. The app does not need this feature to search or export papers.")
        summarize_button = gr.Button("Draft cautious literature snapshot")
        ai_output = gr.Markdown("Search for papers first. Configure the optional AI provider in Render to enable synthesis.")
    with gr.Accordion("Export research", open=False):
        with gr.Row():
            csv_button = gr.Button("Download CSV")
            bib_button = gr.Button("Download BibTeX")
        csv_output = gr.File(label="CSV file")
        bib_output = gr.File(label="BibTeX file")
    with gr.Accordion("Saved library, search history and health", open=False):
        with gr.Row():
            db_button = gr.Button("Check Neon")
            history_button = gr.Button("Load recent searches")
            library_button = gr.Button("Load saved library")
        db_output = gr.Markdown("")
        history_output = gr.Markdown("")
        library_output = gr.Markdown("")
    gr.Markdown("**Research integrity:** Metadata and abstracts are supplied by arXiv. Analytics are descriptive, not a scientific quality score. Verify methods and conclusions in original papers. External APIs and hosting providers can occasionally be unavailable.")

    def run_ui(q, f, s, n):
        papers, message, _elapsed = perform_search(q, f, s, n)
        choices = [(f"{i+1}. {p.title[:115]}", p.url) for i, p in enumerate(papers)]
        return papers, message, render_papers(papers), build_analysis(papers, q), gr.update(choices=choices, value=[]), [], "Select at least two papers.", "", "", ""

    search_outputs = [papers_state, status, papers_html, analytics, paper_choices, chosen_state, comparison, bookmark_status, ai_output, csv_output]
    search_button.click(run_ui, inputs=[query, field, sort, limit], outputs=search_outputs)
    query.submit(run_ui, inputs=[query, field, sort, limit], outputs=search_outputs)

    def update_selected(values):
        return values or []
    paper_choices.change(update_selected, inputs=paper_choices, outputs=chosen_state)

    compare_button.click(compare_papers, inputs=[papers_state, chosen_state], outputs=comparison)
    save_button.click(bookmarks_add, inputs=[papers_state, chosen_state], outputs=bookmark_status)
    summarize_button.click(ai_summary, inputs=[papers_state, query], outputs=ai_output)
    csv_button.click(export_csv, inputs=papers_state, outputs=csv_output)
    bib_button.click(export_bibtex, inputs=papers_state, outputs=bib_output)
    db_button.click(database_status, outputs=db_output)
    history_button.click(history_text, outputs=history_output)
    library_button.click(load_bookmarks, outputs=library_output)

    def reset_ui():
        return ("", [], "Workspace cleared. Enter a topic to begin.", render_papers([]),
                "A quantitative overview appears after a successful search.", gr.update(choices=[], value=[]),
                [], "Select at least two papers.", "", "", None, None)
    clear_button.click(reset_ui, outputs=[query, papers_state, status, papers_html, analytics, paper_choices, chosen_state, comparison, bookmark_status, ai_output, csv_output, bib_output])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
