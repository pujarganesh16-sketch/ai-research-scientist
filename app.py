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
OPENALEX_API = "https://api.openalex.org/works"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()
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
    """Validate external source URLs (kept under this name for DB compatibility)."""
    parsed = urlparse(value or "")
    allowed = {"openalex.org", "doi.org", "www.doi.org", "arxiv.org", "export.arxiv.org", "pubmed.ncbi.nlm.nih.gov"}
    if parsed.scheme == "https" and (parsed.netloc in allowed or parsed.netloc.endswith(".openalex.org")):
        return value
    return "https://openalex.org/"


def normalize_query(value: str) -> str:
    value = " ".join((value or "").replace("\x00", " ").split())
    if len(value) > MAX_QUERY_CHARS:
        raise ValueError(f"Please keep the search query under {MAX_QUERY_CHARS} characters.")
    return value


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not isinstance(inverted_index, dict):
        return ""
    positions = {}
    for word, indexes in inverted_index.items():
        for index in indexes if isinstance(indexes, list) else []:
            if isinstance(index, int) and 0 <= index < 100000:
                positions[index] = word
    return " ".join(positions[i] for i in sorted(positions))[:MAX_ABSTRACT_CHARS]


def parse_openalex(payload: dict[str, Any]) -> list[Paper]:
    papers: list[Paper] = []
    for item in (payload.get("results") or []):
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "Untitled paper").split())
        authors = []
        for authorship in item.get("authorships") or []:
            author = (authorship.get("author") or {}).get("display_name")
            if author:
                authors.append(str(author))
        date = str(item.get("publication_date") or "")[:10]
        updated = str(item.get("updated_date") or "")[:10]
        landing = (item.get("primary_location") or {}).get("landing_page_url")
        doi_raw = item.get("doi") or ""
        doi = str(doi_raw).replace("https://doi.org/", "") if doi_raw else ""
        url = landing or (f"https://doi.org/{doi}" if doi else item.get("id") or "https://openalex.org/")
        categories = []
        for concept in (item.get("topics") or [])[:8]:
            name = concept.get("display_name") if isinstance(concept, dict) else None
            if name:
                categories.append(str(name))
        if not categories:
            for concept in (item.get("concepts") or [])[:5]:
                name = concept.get("display_name") if isinstance(concept, dict) else None
                if name:
                    categories.append(str(name))
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        if title:
            papers.append(Paper(title, tuple(authors), abstract, safe_arxiv_url(str(url)), date, updated, tuple(categories), doi))
    return papers


def openalex_search_params(query: str, field: str, sort: str, limit: int) -> dict[str, Any]:
    params: dict[str, Any] = {"search": query, "per-page": limit, "mailto": os.getenv("CONTACT_EMAIL", "")}
    if OPENALEX_API_KEY:
        params["api_key"] = OPENALEX_API_KEY
    if field == "Title":
        params["filter"] = "title.search:" + query
        params.pop("search", None)
    elif field == "Author":
        params["filter"] = "authorships.author.display_name.search:" + query
        params.pop("search", None)
    if sort == "Newest first":
        params["sort"] = "publication_date:desc"
    return {k: v for k, v in params.items() if v not in (None, "")}


def request_openalex(params: dict[str, Any]) -> dict[str, Any]:
    response = SESSION.get(OPENALEX_API, params=params, timeout=(8, 40))
    response.raise_for_status()
    if len(response.content) > 15_000_000:
        raise RuntimeError("The OpenAlex response was unexpectedly large.")
    return response.json()


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


def perform_search(query: str, field: str, sort: str, limit: int):
    start = time.monotonic()
    query = normalize_query(query)
    if not query:
        return [], "Enter a topic to start your research.", 0
    try:
        limit = max(1, min(int(limit), MAX_RESULTS))
    except (ValueError, TypeError):
        limit = 10
    params = openalex_search_params(query, field, sort, limit)
    try:
        payload = request_openalex(params)
        papers = parse_openalex(payload)
        elapsed = int((time.monotonic() - start) * 1000)
        save_search(query, field, len(papers), elapsed)
        if papers:
            status = f"**{len(papers)} papers found** · {elapsed / 1000:.1f}s · Source: OpenAlex · Search field: {escape(field)}"
        else:
            status = "No papers found. Try shorter or broader keywords."
        log.info("OpenAlex search completed: field=%s count=%d elapsed_ms=%d", field, len(papers), elapsed)
        return papers, status, elapsed
    except requests.Timeout:
        log.warning("OpenAlex timed out")
        return [], "OpenAlex timed out. Wait a moment and retry; the external service may be busy.", 0
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        log.warning("OpenAlex HTTP error: %s", code)
        if code == 429:
            message = "OpenAlex is rate-limiting requests. Please wait briefly before trying again."
        elif code in (401, 403):
            message = "OpenAlex rejected the API key. Check OPENALEX_API_KEY in Render Environment."
        else:
            message = f"OpenAlex returned an HTTP error ({code}). Please retry later."
        return [], message, 0
    except requests.RequestException:
        log.exception("Network error contacting OpenAlex")
        return [], "Could not reach OpenAlex from the server. Check Render logs and try again later.", 0
    except (ValueError, json.JSONDecodeError):
        log.exception("Invalid JSON received from OpenAlex")
        return [], "OpenAlex returned data that could not be read. Please retry later.", 0
    except ValueError as exc:
        return [], escape(str(exc)), 0
    except Exception:
        log.exception("Unexpected search failure")
        return [], "An unexpected search error occurred. Check Render logs for details.", 0


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
        f"**Frequent research topics**\n\n{top_cats}\n\n"
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
        entries.append(f"@misc{{{key},\n  title = {{{b(p.title)}}},\n  author = {{{b(' and '.join(p.authors))}}},\n  year = {{{year}}},\n  howpublished = {{OpenAlex}},\n  url = {{{safe_arxiv_url(p.url)}}}\n}}")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n\n".join(entries) + "\n")
    return path




def export_json(papers: list[Paper]):
    """Export normalized paper metadata as a portable JSON file."""
    if not papers:
        raise gr.Error("Search for papers before exporting.")
    path = os.path.join(tempfile.gettempdir(), f"research_data_{int(time.time())}.json")
    payload = [{
        "title": p.title, "authors": list(p.authors), "abstract": p.abstract,
        "published": p.published, "updated": p.updated, "categories": list(p.categories),
        "doi": p.doi, "url": p.url, "source": "OpenAlex"
    } for p in papers]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def export_ris(papers: list[Paper]):
    """Export records in RIS format for reference managers."""
    if not papers:
        raise gr.Error("Search for papers before exporting.")
    path = os.path.join(tempfile.gettempdir(), f"research_refs_{int(time.time())}.ris")
    rows = []
    for p in papers:
        lines = ["TY  - JOUR", f"TI  - {p.title}"]
        lines.extend(f"AU  - {a}" for a in p.authors)
        if p.published:
            lines.append(f"PY  - {p.published[:4]}")
        if p.doi:
            lines.append(f"DO  - {p.doi}")
        if p.abstract:
            lines.append(f"AB  - {p.abstract.replace(chr(10), ' ')}")
        if p.url:
            lines.append(f"UR  - {p.url}")
        lines.extend(["DB  - OpenAlex", "ER  - ", ""])
        rows.append("\n".join(lines))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows))
    return path


def source_health() -> str:
    """Small live diagnostic for the configured scholarly API."""
    started = time.perf_counter()
    try:
        payload = request_openalex({"search": "machine learning", "per-page": 1, "select": "id,display_name"})
        count = len(payload.get("results", []))
        ms = int((time.perf_counter() - started) * 1000)
        return (f"### OpenAlex: reachable\n\n- Test records returned: **{count}**\n"
                f"- Response time: **{ms} ms**\n- API key configured: **{'yes' if OPENALEX_API_KEY else 'not set'}**")
    except Exception as exc:
        log.exception("OpenAlex health check failed")
        return ("### OpenAlex: unavailable\n\n"
                f"The test request failed with **{escape(type(exc).__name__)}**. "
                "Check the newest Render logs; do not share API keys or database credentials.")


def quick_topic(topic: str):
    """Populate a safe, useful example query from a topic chip."""
    return topic


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
    # Bound input and result size; only summarize abstracts actually returned by OpenAlex.
    source = "\n\n".join(f"TITLE: {p.title}\nABSTRACT: {p.abstract[:1800]}\nSOURCE: {safe_arxiv_url(p.url)}" for p in papers[:8])
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=35, max_retries=1)
        result = client.responses.create(
            model=OPENAI_MODEL,
            input=(
                "You are a careful research assistant. Summarize only evidence in the supplied paper titles and abstracts. "
                "Do not invent methods, results, citations, or consensus. Clearly label uncertainty and missing details. "
                "Use sections: Common themes, Differences, Evidence limitations, Follow-up questions. "
                "Cite claims inline using the supplied source URLs. If abstracts do not support a claim, say so.\n\n"
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
:root{--ars-ink:#172033;--ars-muted:#667085;--ars-purple:#6941c6;--ars-pink:#e94f9b;--ars-cyan:#19b8c7;--ars-border:#e7e9f4}
.gradio-container{max-width:1320px!important;margin:auto!important;padding:24px clamp(12px,3vw,34px) 42px!important;background:radial-gradient(ellipse at 4% 0%,rgba(231,79,155,.08),transparent 34%),radial-gradient(ellipse at 96% 9%,rgba(25,184,199,.10),transparent 32%),var(--background-fill-primary)}
#hero{position:relative;isolation:isolate;overflow:hidden;padding:clamp(28px,5vw,58px);border-radius:30px;background:linear-gradient(120deg,#201447 0%,#5032a4 43%,#8b3ec9 70%,#e94f9b 110%);color:#fff;margin:0 0 24px;box-shadow:0 24px 70px rgba(78,44,151,.23)}
#hero:before,#hero:after{content:"";position:absolute;z-index:-1;border-radius:50%;pointer-events:none}
#hero:before{width:340px;height:340px;right:-70px;top:-170px;background:radial-gradient(circle,rgba(255,255,255,.28),rgba(255,255,255,0) 70%)}
#hero:after{width:280px;height:280px;right:18%;bottom:-225px;border:1px solid #ffffff2e;box-shadow:0 0 0 28px #ffffff09,0 0 0 58px #ffffff07}
#hero .eyebrow{display:inline-flex;color:#f4eaff;text-transform:uppercase;letter-spacing:2.1px;font-size:11px;font-weight:850;margin:0 0 16px;padding:8px 12px;border:1px solid #ffffff45;background:#ffffff12;border-radius:999px}
#hero h1{color:#fff;font-size:clamp(34px,5.5vw,58px);line-height:1.04;margin:0 0 16px;letter-spacing:-1.7px;font-weight:850}
#hero p{color:#f4efff;line-height:1.85;max-width:820px;margin:0;font-size:clamp(14px,1.5vw,17px)}
#hero .hero-meta{display:flex;flex-wrap:wrap;gap:9px;margin-top:26px}
#hero .hero-meta span{font-size:11px;color:white;background:#ffffff16;border:1px solid #ffffff35;border-radius:999px;padding:9px 12px;backdrop-filter:blur(8px)}
.search-panel{border:1px solid var(--ars-border)!important;border-radius:24px!important;padding:clamp(16px,3vw,27px)!important;background:var(--background-fill-primary)!important;box-shadow:0 14px 40px rgba(37,35,84,.06)!important}
.gradio-container button{border-radius:13px!important;font-weight:750!important;transition:transform .16s ease,box-shadow .16s ease!important}
.gradio-container button:hover{transform:translateY(-1px)}
.gradio-container button.primary{background:linear-gradient(100deg,#6841c6,#a13acb 58%,#e94f9b)!important;border:0!important;box-shadow:0 9px 22px rgba(127,60,184,.22)!important}
.gradio-container input,.gradio-container textarea,.gradio-container select{border-radius:12px!important}
.gradio-container input:focus,.gradio-container textarea:focus{border-color:#9b67e8!important;box-shadow:0 0 0 3px rgba(155,103,232,.14)!important}
.gradio-container .accordion{border:1px solid var(--ars-border)!important;border-radius:18px!important;margin-top:14px!important;overflow:hidden}
.papers-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:14px}
.paper-card{min-width:0;padding:clamp(18px,2vw,25px);border:1px solid var(--ars-border);border-radius:22px;background:var(--background-fill-primary);box-shadow:0 9px 28px rgba(35,30,75,.055);transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease}
.paper-card:hover{transform:translateY(-3px);border-color:#c9b5f4;box-shadow:0 18px 42px rgba(91,57,151,.12)}
.paper-top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:15px}
.index{font-size:10px;font-weight:900;letter-spacing:1.2px;color:#7544c7;background:#f2eaff;padding:6px 9px;border-radius:999px}
.date{font-size:11px;color:var(--body-text-color-subdued)}
.paper-card h3{font-size:clamp(16px,1.7vw,19px);line-height:1.5;margin:0 0 12px;color:var(--body-text-color);letter-spacing:-.25px}
.authors{font-size:12px;line-height:1.75;color:var(--body-text-color-subdued);margin-bottom:14px}
.tag-row{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:16px}
.tag{font-size:10px;border-radius:999px;padding:6px 9px;color:#5140a2;background:linear-gradient(100deg,#f1eaff,#e8faff);border:1px solid #e7dcff}
.paper-card details{border-top:1px solid var(--border-color-primary);padding-top:13px}
.paper-card summary{cursor:pointer;font-size:13px;font-weight:800;color:#8150d2;display:flex;justify-content:space-between;gap:8px}
.abstract{font-size:13px;line-height:1.9;margin-top:13px;color:var(--body-text-color)}
.paper-bottom{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap;margin-top:19px}
.paper-bottom a{background:linear-gradient(100deg,#6841c6,#a13acb 65%,#e94f9b);color:#fff!important;text-decoration:none!important;border-radius:12px;padding:11px 15px;font-size:12px;font-weight:850;box-shadow:0 7px 17px rgba(111,64,182,.17)}
.paper-bottom a:hover{filter:brightness(1.07)}
.doi{font-size:10px;color:var(--body-text-color-subdued);overflow-wrap:anywhere}
.empty-state{text-align:center;padding:clamp(32px,6vw,60px) 20px;border:1px dashed #cfc5eb;border-radius:24px;background:linear-gradient(135deg,#faf7ff,#f2fbff)}
.empty-symbol{font-size:42px;color:#8c55d9}.empty-state h3{margin:12px 0;color:#35215d}.empty-state p{color:#667085;line-height:1.8;max-width:600px;margin:0 auto}
footer{visibility:hidden}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;scroll-behavior:auto!important}}
@media(max-width:800px){.gradio-container{padding:12px 10px 28px!important}.papers-grid{grid-template-columns:1fr;gap:13px}#hero{padding:28px 21px;border-radius:23px}.paper-card{padding:18px}.search-panel{border-radius:19px!important}}
"""

with gr.Blocks(title=f"{APP_NAME} | Premium Research Workspace") as demo:
    papers_state = gr.State([])
    chosen_state = gr.State([])
    gr.HTML(
        '<header id="hero"><p class="eyebrow">Academic intelligence workspace · v3.0</p>'
        '<h1>AI Research Scientist</h1>'
        '<p>Explore scholarly literature, inspect source abstracts, compare studies, quantify the research landscape, and export citations in a polished workspace.</p>'
        '<div class="hero-meta"><span>OpenAlex discovery</span><span>Research analytics</span><span>CSV + BibTeX</span><span>Optional Neon library</span></div></header>'
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
        gr.Markdown("Export your current results for spreadsheets, reference managers, or reproducible analysis.")
        with gr.Row():
            csv_button = gr.Button("Download CSV")
            bib_button = gr.Button("Download BibTeX")
            ris_button = gr.Button("Download RIS")
            json_button = gr.Button("Download JSON")
        csv_output = gr.File(label="CSV file")
        bib_output = gr.File(label="BibTeX file")
        ris_output = gr.File(label="RIS file · Zotero / EndNote")
        json_output = gr.File(label="JSON file · structured data")
    with gr.Accordion("API diagnostics", open=False):
        gr.Markdown("Check whether the live app can reach OpenAlex. This test does not expose your API key.")
        health_button = gr.Button("Test OpenAlex connection")
        health_output = gr.Markdown("Connection has not been tested yet.")
    with gr.Accordion("Try a research topic", open=False):
        gr.Markdown("Choose a starter topic, then press Explore papers.")
        with gr.Row():
            topic_ai = gr.Button("Generative AI")
            topic_climate = gr.Button("Climate change")
            topic_health = gr.Button("AI in healthcare")
            topic_robotics = gr.Button("Robotics")
        with gr.Row():
            topic_security = gr.Button("Cybersecurity")
            topic_energy = gr.Button("Renewable energy")
            topic_nlp = gr.Button("Natural language processing")
            topic_bio = gr.Button("Bioinformatics")
    with gr.Accordion("Saved library, search history and health", open=False):
        with gr.Row():
            db_button = gr.Button("Check Neon")
            history_button = gr.Button("Load recent searches")
            library_button = gr.Button("Load saved library")
        db_output = gr.Markdown("")
        history_output = gr.Markdown("")
        library_output = gr.Markdown("")
    gr.Markdown("**Research integrity:** Metadata and abstracts are supplied by OpenAlex; abstracts are not available for every record. Analytics are descriptive, not a scientific quality score. Verify methods and conclusions in original papers. External APIs and hosting providers can occasionally be unavailable.")

    def run_ui(q, f, s, n):
        papers, message, _elapsed = perform_search(q, f, s, n)
        choices = [(f"{i+1}. {p.title[:115]}", p.url) for i, p in enumerate(papers)]
        return papers, message, render_papers(papers), build_analysis(papers, q), gr.update(choices=choices, value=[]), [], "Select at least two papers.", "", "", None

    search_outputs = [papers_state, status, papers_html, analytics, paper_choices, chosen_state, comparison, bookmark_status, ai_output, csv_output]
    search_button.click(run_ui, inputs=[query, field, sort, limit], outputs=search_outputs)
    query.submit(run_ui, inputs=[query, field, sort, limit], outputs=search_outputs)

    def update_selected(values):
        return values or []
    paper_choices.change(update_selected, inputs=paper_choices, outputs=chosen_state)

    compare_button.click(compare_papers, inputs=[papers_state, chosen_state], outputs=comparison)
    save_button.click(bookmarks_add, inputs=[papers_state, chosen_state], outputs=bookmark_status)
    summarize_button.click(lambda papers, q: safe_ai_summary(papers, q), inputs=[papers_state, query], outputs=ai_output)
    csv_button.click(export_csv, inputs=papers_state, outputs=csv_output)
    bib_button.click(export_bibtex, inputs=papers_state, outputs=bib_output)
    ris_button.click(export_ris, inputs=papers_state, outputs=ris_output)
    json_button.click(export_json, inputs=papers_state, outputs=json_output)
    health_button.click(source_health, outputs=health_output)
    for topic_button, topic_value in [
        (topic_ai, "generative AI"), (topic_climate, "climate change"),
        (topic_health, "artificial intelligence healthcare"), (topic_robotics, "robotics"),
        (topic_security, "cybersecurity"), (topic_energy, "renewable energy"),
        (topic_nlp, "natural language processing"), (topic_bio, "bioinformatics")]:
        topic_button.click(lambda value=topic_value: value, outputs=query)
    db_button.click(database_status, outputs=db_output)
    history_button.click(history_text, outputs=history_output)
    library_button.click(load_bookmarks, outputs=library_output)

    def reset_ui():
        return ("", [], "Workspace cleared. Enter a topic to begin.", render_papers([]),
                "A quantitative overview appears after a successful search.", gr.update(choices=[], value=[]),
                [], "Select at least two papers.", "", "", None, None, None, None)
    clear_button.click(reset_ui, outputs=[query, papers_state, status, papers_html, analytics, paper_choices, chosen_state, comparison, bookmark_status, ai_output, csv_output, bib_output, ris_output, json_output])


# ---------------------------------------------------------------------------
# Research intelligence utilities
# ---------------------------------------------------------------------------
# These helpers are deliberately pure (no network or database calls). Keeping
# analysis functions deterministic makes them easier to test and safer to use
# inside a queued web application.


def normalize_whitespace(value: Any) -> str:
    """Return a stable, printable, whitespace-normalized string."""
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(value: Any) -> str:
    """Normalize a DOI from bare, doi.org, or doi: prefixed forms."""
    doi = normalize_whitespace(value)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.strip().rstrip(".,;)")


def doi_url(value: Any) -> str:
    """Return a safe canonical DOI URL, or an empty string when invalid."""
    doi = normalize_doi(value)
    if not doi or not re.match(r"^10\.\d{4,9}/\S+$", doi, flags=re.I):
        return ""
    if any(ch in doi for ch in "<>\"' "):
        return ""
    from urllib.parse import quote
    return "https://doi.org/" + quote(doi, safe="/().;:_-")


def paper_fingerprint(paper: Paper) -> str:
    """Build a stable duplicate-detection key, preferring a DOI when present."""
    doi = normalize_doi(paper.doi).casefold()
    if doi:
        return "doi:" + doi
    title = normalize_whitespace(paper.title).casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title).strip()
    year = str(paper.year or "")
    return f"title:{title}|year:{year}"


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """Preserve result order while dropping records with the same fingerprint."""
    seen: set[str] = set()
    unique: list[Paper] = []
    for paper in papers:
        key = paper_fingerprint(paper)
        if key in seen:
            continue
        seen.add(key)
        unique.append(paper)
    return unique


def paper_completeness(paper: Paper) -> dict[str, bool]:
    """Report which useful bibliographic fields are available for a record."""
    return {
        "title": bool(normalize_whitespace(paper.title)),
        "authors": bool(paper.authors),
        "abstract": bool(normalize_whitespace(paper.abstract)),
        "publication_date": bool(paper.published),
        "source_url": bool(safe_arxiv_url(paper.url)),
        "doi": bool(normalize_doi(paper.doi)),
        "topics": bool(paper.categories),
    }


def completeness_score(paper: Paper) -> float:
    """Calculate a transparent metadata completeness percentage (not quality)."""
    fields = paper_completeness(paper)
    return round(100 * sum(fields.values()) / max(1, len(fields)), 1)


def average_completeness(papers: list[Paper]) -> float:
    """Average metadata completeness across a result set."""
    if not papers:
        return 0.0
    return round(sum(completeness_score(p) for p in papers) / len(papers), 1)


def reading_minutes(text: Any, words_per_minute: int = 220) -> int:
    """Estimate reading time from abstract length; this is only a rough guide."""
    safe_wpm = max(80, min(int(words_per_minute or 220), 600))
    words = len(normalize_whitespace(text).split())
    return max(1, math.ceil(words / safe_wpm)) if words else 0


def abstract_preview(text: Any, limit: int = 280) -> str:
    """Create a plain-text preview without cutting in the middle of a word."""
    text = normalize_whitespace(text)
    limit = max(40, min(int(limit or 280), 2000))
    if len(text) <= limit:
        return text
    shortened = text[: limit + 1]
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return shortened.rstrip(" .,:;-") + "…"


def title_words(paper: Paper) -> set[str]:
    """Extract useful title tokens for transparent overlap comparisons."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", paper.title.lower())
    return {token for token in tokens if token not in STOPWORDS}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Compute Jaccard similarity for two token sets, in the range 0..1."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def title_overlap_matrix(papers: list[Paper]) -> list[list[float]]:
    """Return pairwise title-token overlap; diagonal entries are 1 for nonempty titles."""
    token_sets = [title_words(paper) for paper in papers]
    matrix: list[list[float]] = []
    for i, left in enumerate(token_sets):
        row = []
        for j, right in enumerate(token_sets):
            row.append(1.0 if i == j and left else jaccard_similarity(left, right))
        matrix.append(row)
    return matrix


def year_distribution(papers: list[Paper]) -> list[tuple[int, int]]:
    """Count publication years and return them in chronological order."""
    counts = Counter(p.year for p in papers if p.year is not None)
    return sorted(counts.items())


def author_productivity(papers: list[Paper], limit: int = 10) -> list[tuple[str, int]]:
    """Count authors within the current result set, not across all scholarship."""
    counts = Counter(author for paper in papers for author in paper.authors if author)
    return counts.most_common(max(1, min(int(limit or 10), 50)))


def topic_distribution(papers: list[Paper], limit: int = 12) -> list[tuple[str, int]]:
    """Count source-provided topics/categories in the current result set."""
    counts = Counter(category for paper in papers for category in paper.categories if category)
    return counts.most_common(max(1, min(int(limit or 12), 50)))


def abstract_coverage(papers: list[Paper]) -> float:
    """Return the percentage of records that include an abstract."""
    if not papers:
        return 0.0
    return round(100 * sum(bool(p.abstract.strip()) for p in papers) / len(papers), 1)


def doi_coverage(papers: list[Paper]) -> float:
    """Return the percentage of records with a DOI value."""
    if not papers:
        return 0.0
    return round(100 * sum(bool(normalize_doi(p.doi)) for p in papers) / len(papers), 1)


def date_coverage(papers: list[Paper]) -> float:
    """Return the percentage of records with a four-digit publication year."""
    if not papers:
        return 0.0
    return round(100 * sum(p.year is not None for p in papers) / len(papers), 1)


def result_set_summary(papers: list[Paper]) -> dict[str, Any]:
    """Produce a compact, serializable overview for dashboard components."""
    years = [year for year, _count in year_distribution(papers)]
    return {
        "count": len(papers),
        "unique_authors": len({a for p in papers for a in p.authors}),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "abstract_coverage_percent": abstract_coverage(papers),
        "doi_coverage_percent": doi_coverage(papers),
        "date_coverage_percent": date_coverage(papers),
        "metadata_completeness_percent": average_completeness(papers),
        "year_distribution": year_distribution(papers),
        "top_authors": author_productivity(papers, 8),
        "top_topics": topic_distribution(papers, 8),
    }


def filter_by_year(papers: list[Paper], start_year: int | None = None,
                   end_year: int | None = None) -> list[Paper]:
    """Filter a result list by inclusive publication-year bounds."""
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("Start year must be less than or equal to end year.")
    filtered = []
    for paper in papers:
        year = paper.year
        if year is None:
            continue
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        filtered.append(paper)
    return filtered


def filter_by_author(papers: list[Paper], fragment: str) -> list[Paper]:
    """Case-insensitive author substring filter for an already-fetched result set."""
    needle = normalize_whitespace(fragment).casefold()
    if not needle:
        return list(papers)
    return [p for p in papers if any(needle in author.casefold() for author in p.authors)]


def filter_by_topic(papers: list[Paper], fragment: str) -> list[Paper]:
    """Case-insensitive category/topic substring filter."""
    needle = normalize_whitespace(fragment).casefold()
    if not needle:
        return list(papers)
    return [p for p in papers if any(needle in topic.casefold() for topic in p.categories)]


def filter_with_abstract(papers: list[Paper], required: bool = True) -> list[Paper]:
    """Keep records with or without abstracts, depending on the requested mode."""
    return [p for p in papers if bool(p.abstract.strip()) is bool(required)]


def sort_papers_locally(papers: list[Paper], mode: str) -> list[Paper]:
    """Sort already-fetched results without another API request."""
    if mode == "Newest first":
        return sorted(papers, key=lambda p: (p.published or "", p.title.casefold()), reverse=True)
    if mode == "Title A–Z":
        return sorted(papers, key=lambda p: p.title.casefold())
    if mode == "Most authors":
        return sorted(papers, key=lambda p: (len(p.authors), p.title.casefold()), reverse=True)
    if mode == "Abstract available":
        return sorted(papers, key=lambda p: (bool(p.abstract), len(p.abstract)), reverse=True)
    return list(papers)


def search_text_matches(paper: Paper, query: str) -> bool:
    """Check whether every query token appears in available local paper text."""
    terms = [term.casefold() for term in normalize_whitespace(query).split()]
    haystack = " ".join((paper.title, paper.abstract, paper.author_text,
                         " ".join(paper.categories))).casefold()
    return all(term in haystack for term in terms)


def local_text_filter(papers: list[Paper], query: str) -> list[Paper]:
    """Filter already-fetched papers without implying full-database coverage."""
    if not normalize_whitespace(query):
        return list(papers)
    return [paper for paper in papers if search_text_matches(paper, query)]


def citation_author_label(authors: tuple[str, ...], maximum: int = 3) -> str:
    """Return a compact author label suitable for an informal citation preview."""
    names = [normalize_whitespace(a) for a in authors if normalize_whitespace(a)]
    if not names:
        return "Unknown author"
    if len(names) <= maximum:
        return ", ".join(names)
    return ", ".join(names[:maximum]) + ", et al."


def citation_year_label(paper: Paper) -> str:
    """Return a year label that does not invent missing publication dates."""
    return str(paper.year) if paper.year is not None else "n.d."


def apa_like_reference(paper: Paper) -> str:
    """Build a readable APA-like reference preview; not a formal style validator."""
    author = citation_author_label(paper.authors)
    year = citation_year_label(paper)
    title = normalize_whitespace(paper.title).rstrip(".")
    doi = doi_url(paper.doi)
    base = f"{author} ({year}). {title}."
    return f"{base} {doi}" if doi else base


def markdown_reference_list(papers: list[Paper]) -> str:
    """Render a Markdown bibliography with escaped user-visible content."""
    if not papers:
        return "No papers are available to cite yet."
    return "\n\n".join(f"{index}. {escape(apa_like_reference(paper))}"
                          for index, paper in enumerate(papers, 1))


def keyword_cooccurrence(papers: list[Paper], top_n: int = 20) -> list[tuple[str, str, int]]:
    """Count co-occurring frequent title terms within the same paper."""
    per_paper = []
    global_counts = Counter()
    for paper in papers:
        words = title_words(paper)
        per_paper.append(words)
        global_counts.update(words)
    selected = {word for word, _count in global_counts.most_common(max(2, top_n))}
    pairs: Counter[tuple[str, str]] = Counter()
    for words in per_paper:
        ordered = sorted(words & selected)
        for i, first in enumerate(ordered):
            for second in ordered[i + 1:]:
                pairs[(first, second)] += 1
    return [(a, b, count) for (a, b), count in pairs.most_common(50)]


def author_collaboration_edges(papers: list[Paper], maximum_authors_per_paper: int = 12) -> list[tuple[str, str, int]]:
    """Summarize co-authorship edges observed in the current result set."""
    edges: Counter[tuple[str, str]] = Counter()
    for paper in papers:
        authors = sorted(set(paper.authors[:max(2, maximum_authors_per_paper)]))
        for index, first in enumerate(authors):
            for second in authors[index + 1:]:
                edges[(first, second)] += 1
    return [(a, b, n) for (a, b), n in edges.most_common(100)]


def result_quality_notes(papers: list[Paper]) -> list[str]:
    """Generate cautious metadata observations, never quality judgments."""
    if not papers:
        return ["Run a search before evaluating metadata coverage."]
    notes = []
    abstract_pct = abstract_coverage(papers)
    doi_pct = doi_coverage(papers)
    date_pct = date_coverage(papers)
    if abstract_pct < 50:
        notes.append("Fewer than half of these records include abstracts; inspect publisher pages for context.")
    elif abstract_pct < 90:
        notes.append("Some records lack abstracts, so text-based comparisons may be incomplete.")
    else:
        notes.append("Most records include abstracts, supporting a more informative first-pass review.")
    if doi_pct < 50:
        notes.append("DOI coverage is limited in this result set; use source links to verify individual records.")
    if date_pct < 90:
        notes.append("Some publication dates are unavailable; year-based analysis excludes undated records.")
    notes.append("Metadata completeness is not a measure of scientific rigor or trustworthiness.")
    return notes


def safe_filename(label: str, extension: str, max_stem: int = 64) -> str:
    """Create a conservative filename for generated downloads."""
    stem = normalize_whitespace(label).lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    stem = (stem or "research-export")[:max(12, min(max_stem, 120))].strip("-")
    ext = re.sub(r"[^a-z0-9]", "", (extension or "txt").lower())[:8] or "txt"
    return f"{stem}.{ext}"


def bounded_integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    """Parse an integer input and clamp it to an explicit safe range."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(minimum, min(maximum, number))


def parse_year_input(value: Any) -> int | None:
    """Parse a year-like input; blank values mean no bound."""
    if value is None or normalize_whitespace(value) == "":
        return None
    year = bounded_integer(value, -1, 1000, 2200)
    if year == -1:
        raise ValueError("Enter a valid year between 1000 and 2200.")
    return year


def validate_result_limit(value: Any) -> int:
    """Normalize user-provided result counts to the app's supported range."""
    return bounded_integer(value, 10, 1, MAX_RESULTS)


def safe_markdown_text(value: Any) -> str:
    """Escape raw HTML and normalize line breaks for a plain Markdown label."""
    text = normalize_whitespace(value)
    return escape(text).replace("|", "\\|")


def compact_paper_record(paper: Paper) -> dict[str, Any]:
    """Serialize a Paper to a stable, JSON-compatible public record."""
    return {
        "title": paper.title,
        "authors": list(paper.authors),
        "abstract": paper.abstract,
        "url": safe_arxiv_url(paper.url),
        "published": paper.published,
        "updated": paper.updated,
        "categories": list(paper.categories),
        "doi": normalize_doi(paper.doi),
        "year": paper.year,
        "metadata_completeness_percent": completeness_score(paper),
    }


def compact_result_records(papers: list[Paper]) -> list[dict[str, Any]]:
    """Serialize a complete result set for downstream analysis and testing."""
    return [compact_paper_record(paper) for paper in papers]


def result_set_signature(papers: list[Paper]) -> str:
    """Return a deterministic short signature for a result set, not a security token."""
    import hashlib
    payload = "\n".join(paper_fingerprint(p) for p in papers).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def compare_metadata(left: Paper, right: Paper) -> dict[str, Any]:
    """Return a transparent side-by-side metadata comparison."""
    return {
        "same_doi": bool(normalize_doi(left.doi)) and normalize_doi(left.doi).casefold() == normalize_doi(right.doi).casefold(),
        "title_similarity": round(jaccard_similarity(title_words(left), title_words(right)), 3),
        "shared_authors": sorted(set(left.authors) & set(right.authors)),
        "shared_topics": sorted(set(left.categories) & set(right.categories)),
        "year_difference": abs(left.year - right.year) if left.year and right.year else None,
        "both_have_abstracts": bool(left.abstract.strip()) and bool(right.abstract.strip()),
    }


def duplicate_groups(papers: list[Paper]) -> list[list[Paper]]:
    """Find groups with matching normalized DOI/title-year fingerprints."""
    groups: dict[str, list[Paper]] = {}
    for paper in papers:
        groups.setdefault(paper_fingerprint(paper), []).append(paper)
    return [group for group in groups.values() if len(group) > 1]


def research_landscape_report(papers: list[Paper], query: str = "") -> str:
    """Create a plain-text summary suitable for download or copy/paste."""
    summary = result_set_summary(papers)
    lines = [f"Research landscape report: {normalize_whitespace(query) or 'Unspecified topic'}",
             "=" * 68,
             f"Records in current result set: {summary['count']}",
             f"Distinct listed authors: {summary['unique_authors']}",
             f"Abstract coverage: {summary['abstract_coverage_percent']}%",
             f"DOI coverage: {summary['doi_coverage_percent']}%",
             f"Metadata completeness: {summary['metadata_completeness_percent']}%"]
    if summary["year_min"] is not None:
        lines.append(f"Publication years represented: {summary['year_min']}–{summary['year_max']}")
    lines.append("")
    lines.append("Interpretation cautions:")
    lines.extend(f"- {note}" for note in result_quality_notes(papers))
    return "\n".join(lines)


def validate_paper_object(paper: Any) -> bool:
    """Validate the expected Paper shape before passing records to renderers."""
    if not isinstance(paper, Paper):
        return False
    if not isinstance(paper.title, str) or not isinstance(paper.authors, tuple):
        return False
    if not isinstance(paper.abstract, str) or not isinstance(paper.url, str):
        return False
    if not isinstance(paper.categories, tuple):
        return False
    return True


def validate_paper_collection(papers: Any) -> list[Paper]:
    """Return only structurally valid Paper objects from an untrusted collection."""
    if not isinstance(papers, (list, tuple)):
        return []
    return [paper for paper in papers if validate_paper_object(paper)]


def merge_paper_collections(*collections: list[Paper]) -> list[Paper]:
    """Merge multiple result collections while preserving first-seen ordering."""
    merged: list[Paper] = []
    for collection in collections:
        merged.extend(validate_paper_collection(collection))
    return deduplicate_papers(merged)


def export_manifest(papers: list[Paper], query: str = "") -> dict[str, Any]:
    """Describe an export with provenance and counts, without secret values."""
    return {
        "application": APP_NAME,
        "application_version": APP_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": normalize_whitespace(query),
        "source": "OpenAlex",
        "record_count": len(papers),
        "result_signature": result_set_signature(papers),
        "records": compact_result_records(papers),
        "limitations": [
            "This export contains only records returned by the current search.",
            "Missing abstracts or dates are not inferred.",
            "Metadata completeness is not a quality score.",
        ],
    }


def search_diagnostics_snapshot() -> dict[str, Any]:
    """Return safe runtime diagnostics without returning credentials or URLs with keys."""
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "python_version": __import__("sys").version.split()[0],
        "requests_version": getattr(requests, "__version__", "unknown"),
        "gradio_version": getattr(gr, "__version__", "unknown"),
        "openalex_key_configured": bool(OPENALEX_API_KEY),
        "database_configured": bool(DATABASE_URL),
        "ai_synthesis_configured": bool(OPENAI_API_KEY and OpenAI is not None),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def format_diagnostics_markdown() -> str:
    """Render a privacy-safe diagnostics summary for troubleshooting."""
    data = search_diagnostics_snapshot()
    rows = ["### Application diagnostics", "", "| Check | Value |", "|---|---|"]
    for key, value in data.items():
        label = key.replace("_", " ").capitalize()
        rows.append(f"| {escape(label)} | {escape(value)} |")
    rows.extend(["", "No API key values, database credentials, or user search history are included."])
    return "\n".join(rows)


def api_error_guidance(status_code: int | None, response_text: str = "") -> str:
    """Map common HTTP status codes to actionable, non-secret guidance."""
    if status_code == 400:
        return "The provider rejected the query. Try a shorter, simpler search phrase."
    if status_code == 401:
        return "The provider did not authenticate the request. Check the server environment variable."
    if status_code == 403:
        return "Access was denied. Verify key activation, account access, and API usage policy."
    if status_code == 404:
        return "The requested provider endpoint was not found. Check the configured API endpoint."
    if status_code == 429:
        return "The provider rate limit was reached. Wait briefly before searching again."
    if status_code in (500, 502, 503, 504):
        return "The provider is temporarily unavailable. Retry in a little while."
    if status_code is None:
        return "The provider could not be reached. Check connectivity and try again."
    return "The provider returned an unexpected response. Check server logs for the request details."


def retry_delay_seconds(attempt: int, base: float = 0.5, maximum: float = 8.0) -> float:
    """Calculate capped exponential backoff for custom retry loops."""
    attempt = bounded_integer(attempt, 0, 0, 12)
    return min(maximum, max(0.0, base) * (2 ** attempt))


def truncate_for_log(value: Any, limit: int = 240) -> str:
    """Make a bounded single-line diagnostic value safe for application logs."""
    text = normalize_whitespace(value)
    text = re.sub(r"(?i)(api_key|authorization|password|secret)=?[^&\s]*", r"\1=[redacted]", text)
    return text[:max(20, min(limit, 1000))]


def build_accessibility_label(paper: Paper) -> str:
    """Create concise accessible text for a paper card or screen-reader label."""
    title = normalize_whitespace(paper.title) or "Untitled paper"
    year = citation_year_label(paper)
    author = citation_author_label(paper.authors, maximum=2)
    abstract_state = "abstract available" if paper.abstract.strip() else "abstract unavailable"
    return f"{title}. {author}. {year}. {abstract_state}."


def reading_queue(papers: list[Paper], max_items: int = 20) -> list[dict[str, Any]]:
    """Create a small reading queue ordered by abstract availability then recency."""
    limit = bounded_integer(max_items, 20, 1, 100)
    ordered = sorted(papers, key=lambda p: (bool(p.abstract), p.published or ""), reverse=True)
    queue = []
    for index, paper in enumerate(ordered[:limit], 1):
        queue.append({
            "position": index,
            "title": paper.title,
            "authors": citation_author_label(paper.authors),
            "year": paper.year,
            "url": safe_arxiv_url(paper.url),
            "estimated_abstract_reading_minutes": reading_minutes(paper.abstract),
            "metadata_completeness_percent": completeness_score(paper),
        })
    return queue


def bibliography_statistics(papers: list[Paper]) -> dict[str, Any]:
    """Summarize bibliographic identifiers and fields available for exports."""
    return {
        "records": len(papers),
        "unique_dois": len({normalize_doi(p.doi).casefold() for p in papers if normalize_doi(p.doi)}),
        "records_without_doi": sum(not normalize_doi(p.doi) for p in papers),
        "records_without_authors": sum(not p.authors for p in papers),
        "records_without_abstract": sum(not p.abstract.strip() for p in papers),
        "records_without_year": sum(p.year is None for p in papers),
    }


def research_diversity_index(papers: list[Paper]) -> float:
    """Calculate a normalized topic diversity score from source topic labels."""
    counts = Counter(topic for paper in papers for topic in paper.categories)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return round(entropy / math.log(len(counts)), 4) if total else 0.0


def author_overlap(left: Paper, right: Paper) -> float:
    """Return Jaccard overlap of author names for two paper records."""
    a, b = set(left.authors), set(right.authors)
    return jaccard_similarity(a, b)


def topic_overlap(left: Paper, right: Paper) -> float:
    """Return Jaccard overlap of source-provided topic labels."""
    return jaccard_similarity(set(left.categories), set(right.categories))


def explain_similarity(left: Paper, right: Paper) -> str:
    """Explain simple metadata overlap without presenting it as semantic similarity."""
    data = compare_metadata(left, right)
    shared_authors = ", ".join(data["shared_authors"]) or "none listed"
    shared_topics = ", ".join(data["shared_topics"]) or "none listed"
    return (f"Title-token overlap: {data['title_similarity']:.0%}. "
            f"Shared listed authors: {shared_authors}. Shared topics: {shared_topics}. "
            "This is a metadata comparison, not a scientific assessment.")


def export_validation_report(papers: list[Paper]) -> list[str]:
    """Check a collection before export and return human-readable warnings."""
    warnings = []
    if not papers:
        warnings.append("The export contains no records.")
        return warnings
    if len(deduplicate_papers(papers)) != len(papers):
        warnings.append("Potential duplicate records were detected; review before citation management.")
    if abstract_coverage(papers) < 100:
        warnings.append("Some records have no abstract; this is common and does not necessarily indicate an error.")
    if doi_coverage(papers) < 100:
        warnings.append("Some records lack a DOI; retain their source URLs for traceability.")
    if date_coverage(papers) < 100:
        warnings.append("Some records have no publication year; year charts exclude those records.")
    return warnings


def paper_to_accessible_summary(paper: Paper) -> str:
    """Build a plain-text, accessible summary of one result record."""
    fields = [
        f"Title: {normalize_whitespace(paper.title) or 'Untitled paper'}",
        f"Authors: {paper.author_text}",
        f"Publication date: {paper.published or 'Not available'}",
        f"Topics: {', '.join(paper.categories) if paper.categories else 'Not supplied'}",
        f"Abstract: {abstract_preview(paper.abstract, 1000) if paper.abstract else 'Not supplied'}",
        f"DOI: {normalize_doi(paper.doi) if paper.doi else 'Not supplied'}",
        f"Source: {safe_arxiv_url(paper.url)}",
    ]
    return "\n".join(fields)


def make_search_audit_record(query: str, field: str, sort: str, limit: Any,
                             count: int, elapsed_ms: int) -> dict[str, Any]:
    """Create a minimal non-sensitive audit record for a search operation."""
    return {
        "query": normalize_query(query),
        "field": normalize_whitespace(field)[:40],
        "sort": normalize_whitespace(sort)[:40],
        "limit": validate_result_limit(limit),
        "result_count": bounded_integer(count, 0, 0, MAX_RESULTS),
        "elapsed_ms": bounded_integer(elapsed_ms, 0, 0, 3_600_000),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def calculate_search_efficiency(count: int, elapsed_ms: int) -> float:
    """Return records per second for diagnostics, rounded to two decimals."""
    if elapsed_ms <= 0 or count <= 0:
        return 0.0
    return round(count / (elapsed_ms / 1000), 2)


def safe_percentage(numerator: int | float, denominator: int | float) -> float:
    """Calculate a percentage while safely handling zero or invalid denominators."""
    try:
        if denominator <= 0:
            return 0.0
        return round(100 * float(numerator) / float(denominator), 1)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def abstract_length_statistics(papers: list[Paper]) -> dict[str, float | int]:
    """Summarize abstract lengths in words without retaining additional text."""
    lengths = [len(p.abstract.split()) for p in papers if p.abstract.strip()]
    if not lengths:
        return {"count": 0, "minimum_words": 0, "maximum_words": 0, "average_words": 0.0, "median_words": 0.0}
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    median = float(ordered[midpoint]) if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "count": len(lengths),
        "minimum_words": min(lengths),
        "maximum_words": max(lengths),
        "average_words": round(sum(lengths) / len(lengths), 1),
        "median_words": round(median, 1),
    }


def topic_year_matrix(papers: list[Paper], top_topics: int = 8) -> dict[str, dict[int, int]]:
    """Count topic labels by publication year for simple trend visualizations."""
    topics = [name for name, _count in topic_distribution(papers, top_topics)]
    matrix: dict[str, dict[int, int]] = {topic: {} for topic in topics}
    for paper in papers:
        if paper.year is None:
            continue
        for topic in set(paper.categories) & set(topics):
            matrix[topic][paper.year] = matrix[topic].get(paper.year, 0) + 1
    return matrix


def safe_external_link(value: Any) -> str:
    """Return an allow-listed HTTPS URL suitable for use in rendered links."""
    return safe_arxiv_url(normalize_whitespace(value))


def normalize_sort_mode(value: Any) -> str:
    """Normalize a sort option against supported local sorting modes."""
    allowed = {"Most relevant", "Newest first", "Title A–Z", "Most authors", "Abstract available"}
    value = normalize_whitespace(value)
    return value if value in allowed else "Most relevant"


def build_export_filename(query: str, extension: str) -> str:
    """Generate a descriptive but safe filename for downloadable research data."""
    return safe_filename(query, extension)


def app_capabilities() -> list[str]:
    """List active feature families without implying optional services are enabled."""
    capabilities = ["OpenAlex scholarly search", "Metadata analytics", "Paper comparison",
                    "CSV export", "BibTeX export", "RIS export", "JSON export"]
    if DATABASE_URL:
        capabilities.append("Optional database-backed bookmarks and history")
    if OPENAI_API_KEY and OpenAI is not None:
        capabilities.append("Optional AI-assisted synthesis")
    return capabilities


def capability_report() -> str:
    """Render available capabilities in a readable, non-secret format."""
    lines = ["### Feature availability", ""]
    lines.extend(f"- {escape(capability)}" for capability in app_capabilities())
    lines.append("\nOptional integrations depend on valid credentials and provider availability.")
    return "\n".join(lines)


def validate_search_inputs(query: Any, field: Any, sort: Any, limit: Any) -> tuple[str, str, str, int]:
    """Validate all public search controls in one reusable, testable place."""
    normalized_query = normalize_query(query)
    if not normalized_query:
        raise ValueError("Enter a research topic before searching.")
    allowed_fields = {"All fields", "Title", "Author", "Abstract"}
    normalized_field = normalize_whitespace(field)
    if normalized_field not in allowed_fields:
        normalized_field = "All fields"
    return normalized_query, normalized_field, normalize_sort_mode(sort), validate_result_limit(limit)


def stable_paper_sort_key(paper: Paper) -> tuple[str, str, str]:
    """Stable secondary key for deterministic local ordering."""
    return (paper.published or "", paper.title.casefold(), normalize_doi(paper.doi).casefold())


def sort_by_publication_date(papers: list[Paper], newest_first: bool = True) -> list[Paper]:
    """Sort by publication date with deterministic title/DOI tie breakers."""
    return sorted(papers, key=stable_paper_sort_key, reverse=bool(newest_first))


def records_with_missing_fields(papers: list[Paper], field_name: str) -> list[Paper]:
    """Select records missing a supported bibliographic field."""
    field_map = {
        "abstract": lambda p: bool(p.abstract.strip()),
        "doi": lambda p: bool(normalize_doi(p.doi)),
        "authors": lambda p: bool(p.authors),
        "year": lambda p: p.year is not None,
        "topics": lambda p: bool(p.categories),
    }
    if field_name not in field_map:
        raise ValueError("Supported fields are abstract, doi, authors, year, and topics.")
    return [paper for paper in papers if not field_map[field_name](paper)]


def deduplicated_author_names(papers: list[Paper]) -> list[str]:
    """Return case-insensitively unique author names in first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for paper in papers:
        for author in paper.authors:
            key = normalize_whitespace(author).casefold()
            if key and key not in seen:
                seen.add(key)
                result.append(normalize_whitespace(author))
    return result


def normalized_category_names(papers: list[Paper]) -> list[str]:
    """Return unique source-provided category names sorted for display."""
    return sorted({normalize_whitespace(c) for p in papers for c in p.categories if normalize_whitespace(c)}, key=str.casefold)


def paper_title_slug(paper: Paper) -> str:
    """Create a short title slug for logs or internal labels."""
    return safe_filename(paper.title, "", max_stem=80).rstrip(".")


def citation_key(paper: Paper) -> str:
    """Generate a deterministic BibTeX-style key from author/year/title."""
    first_author = paper.authors[0].split()[-1] if paper.authors else "unknown"
    year = str(paper.year or "nd")
    title_token = next(iter(title_words(paper)), "paper")
    return re.sub(r"[^A-Za-z0-9_-]", "", f"{first_author}{year}{title_token}")[:80] or "researchpaper"


def reading_priority(paper: Paper) -> tuple[int, int, int]:
    """A transparent queue priority based on metadata availability and recency."""
    return (int(bool(paper.abstract.strip())), int(paper.year or 0), len(paper.categories))


def rank_reading_queue(papers: list[Paper]) -> list[Paper]:
    """Rank papers for convenient first-pass reading, not scientific importance."""
    return sorted(papers, key=reading_priority, reverse=True)


def safe_csv_cell(value: Any) -> str:
    """Reduce spreadsheet formula-injection risk for text exported to CSV."""
    text = normalize_whitespace(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def source_identifier(paper: Paper) -> str:
    """Extract the stable OpenAlex work identifier from a supported OpenAlex URL."""
    url = safe_arxiv_url(paper.url)
    match = re.search(r"openalex\.org/(W\d+)$", url)
    return match.group(1) if match else ""


def paper_metadata_warnings(paper: Paper) -> list[str]:
    """Return field-level warnings for one record before a user cites it."""
    warnings = []
    if not paper.title.strip() or paper.title == "Untitled paper":
        warnings.append("Title is missing or generic.")
    if not paper.authors:
        warnings.append("No authors are listed in this record.")
    if not paper.published:
        warnings.append("Publication date is unavailable.")
    if not paper.abstract.strip():
        warnings.append("Abstract is unavailable in the provider record.")
    if not normalize_doi(paper.doi):
        warnings.append("DOI is not supplied; verify the source URL.")
    return warnings


def collection_health(papers: list[Paper]) -> dict[str, Any]:
    """Summarize duplicate and missing-field signals for a collection."""
    valid = validate_paper_collection(papers)
    duplicates = duplicate_groups(valid)
    missing = {field: len(records_with_missing_fields(valid, field))
               for field in ("abstract", "doi", "authors", "year", "topics")}
    return {
        "valid_record_count": len(valid),
        "duplicate_groups": len(duplicates),
        "duplicate_records": sum(len(group) for group in duplicates),
        "missing_field_counts": missing,
        "metadata_completeness_percent": average_completeness(valid),
    }


def normalize_openalex_id(value: Any) -> str:
    """Validate and normalize an OpenAlex work ID or URL."""
    text = normalize_whitespace(value).rstrip("/")
    match = re.fullmatch(r"(?:https://openalex\.org/)?(W\d+)", text, flags=re.I)
    return match.group(1).upper() if match else ""


def validate_provider_payload(payload: Any) -> tuple[bool, str]:
    """Validate the basic shape of a provider response without raising errors."""
    if not isinstance(payload, dict):
        return False, "The provider response was not a JSON object."
    if "results" not in payload:
        return False, "The provider response did not include a results collection."
    results = payload.get("results")
    if results is None:
        return True, "The provider returned no results."
    if not isinstance(results, list):
        return False, "The provider results field was not a list."
    return True, "Provider response structure looks valid."


def provider_result_count(payload: Any) -> int:
    """Return a safe result count from an OpenAlex-like response."""
    if not isinstance(payload, dict):
        return 0
    results = payload.get("results")
    return len(results) if isinstance(results, list) else 0


def robust_json_dumps(value: Any, indent: int = 2) -> str:
    """Serialize JSON with Unicode preserved and non-finite numbers rejected."""
    return json.dumps(value, ensure_ascii=False, indent=bounded_integer(indent, 2, 0, 8), allow_nan=False, default=str)


def report_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for generated reports."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def research_session_summary(query: str, papers: list[Paper], elapsed_ms: int = 0) -> dict[str, Any]:
    """Combine result summary and runtime into a portable research session object."""
    return {
        "query": normalize_whitespace(query),
        "created_at_utc": report_timestamp(),
        "elapsed_ms": bounded_integer(elapsed_ms, 0, 0, 3_600_000),
        "records_per_second": calculate_search_efficiency(len(papers), elapsed_ms),
        "landscape": result_set_summary(papers),
        "collection_health": collection_health(papers),
        "bibliography": bibliography_statistics(papers),
        "papers": compact_result_records(papers),
    }


# End of research intelligence utilities.



# =============================================================================
# Production reliability, caching, persistence, and research-analysis extension
# =============================================================================
# This extension deliberately remains in app.py so a simple Gradio deployment
# does not require a separate backend module. It uses bounded resources, avoids
# storing credentials, and labels heuristics as heuristics rather than evidence.

import hashlib as _hashlib
import sqlite3 as _sqlite3
import threading as _threading
import random as _random
from pathlib import Path as _Path
from contextlib import contextmanager as _contextmanager

FALLBACK_SUMMARY_WARNING = (
    "AI Summary temporarily unavailable; displaying publisher abstract."
)
WORKSPACE_DB_PATH = os.getenv("WORKSPACE_DB_PATH", "research_workspace.sqlite3")
CACHE_TTL_SECONDS = bounded_integer(os.getenv("CACHE_TTL_SECONDS", "300"), 300, 0, 86400)
CACHE_MAX_ENTRIES = bounded_integer(os.getenv("CACHE_MAX_ENTRIES", "256"), 256, 16, 5000)
_WORKSPACE_LOCK = _threading.RLock()
_MEMORY_CACHE: dict[str, tuple[float, Any]] = {}


def stable_cache_key(namespace: str, payload: Any) -> str:
    """Hash request parameters deterministically so cache keys are repeatable."""
    raw = json.dumps(
        {"namespace": normalize_whitespace(namespace), "payload": payload},
        sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"),
    ).encode("utf-8")
    return _hashlib.sha256(raw).hexdigest()


def cache_get(key: str, default: Any = None) -> Any:
    """Return a cached value or default, transparently expiring stale values."""
    now = time.monotonic()
    with _WORKSPACE_LOCK:
        entry = _MEMORY_CACHE.get(str(key))
        if entry is None:
            return default
        expires_at, value = entry
        if expires_at <= now:
            _MEMORY_CACHE.pop(str(key), None)
            return default
        return value


def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    """Set a process-local cache entry with a bounded lifetime and capacity."""
    duration = CACHE_TTL_SECONDS if ttl is None else bounded_integer(ttl, 0, 0, 86400)
    if duration <= 0:
        return
    now = time.monotonic()
    with _WORKSPACE_LOCK:
        for old_key in [k for k, (expiry, _) in _MEMORY_CACHE.items() if expiry <= now]:
            _MEMORY_CACHE.pop(old_key, None)
        if len(_MEMORY_CACHE) >= CACHE_MAX_ENTRIES and key not in _MEMORY_CACHE:
            oldest = min(_MEMORY_CACHE, key=lambda k: _MEMORY_CACHE[k][0])
            _MEMORY_CACHE.pop(oldest, None)
        _MEMORY_CACHE[str(key)] = (now + duration, value)


def cache_clear() -> int:
    """Clear process-local cache entries and return the number removed."""
    with _WORKSPACE_LOCK:
        count = len(_MEMORY_CACHE)
        _MEMORY_CACHE.clear()
        return count


def cached_call(namespace: str, payload: Any, operation, ttl: int | None = None):
    """Execute an operation on cache miss; exceptions are never cached."""
    key = stable_cache_key(namespace, payload)
    marker = object()
    value = cache_get(key, marker)
    if value is not marker:
        return value, True
    result = operation()
    cache_set(key, result, ttl)
    return result, False


def retry_with_exponential_backoff(
    operation, *, attempts: int = 4, base_delay: float = 0.5,
    max_delay: float = 8.0, jitter: float = 0.2, retry_if=None,
):
    """Retry transient operations with capped exponential delay and random jitter.

    This is for polite resilience to temporary network failures. It must not be
    used to bypass authentication, access controls, CAPTCHAs, or site policies.
    """
    total = bounded_integer(attempts, 4, 1, 8)
    base = max(0.0, min(float(base_delay), 30.0))
    cap = max(base, min(float(max_delay), 60.0))
    spread = max(0.0, min(float(jitter), 1.0))
    last_error = None
    for attempt in range(total):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            retryable = retry_if(exc) if callable(retry_if) else True
            if not retryable or attempt + 1 >= total:
                raise
            delay = min(cap, base * (2 ** attempt))
            time.sleep(delay + _random.uniform(0, delay * spread))
    if last_error:
        raise last_error
    raise RuntimeError("Retry operation completed without a result.")


def transient_http_error(error: Exception) -> bool:
    """Identify transient HTTP statuses and common request transport failures."""
    if isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        return bool(response is not None and (
            response.status_code in (408, 425, 429) or response.status_code >= 500
        ))
    return isinstance(error, (
        requests.exceptions.Timeout, requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ))


def safe_ai_summary(papers: list[Paper], query: str) -> str:
    """Return AI synthesis when available, otherwise show original abstracts."""
    try:
        result = ai_summary(papers, query)
        if isinstance(result, str) and result.strip() and not any(marker in result.lower() for marker in ("ai synthesis failed", "empty response")):
            return result
    except Exception:
        log.exception("AI synthesis failed; falling back to publisher abstracts.")
    sections = []
    for paper in (papers or [])[:8]:
        abstract = normalize_whitespace(getattr(paper, "abstract", ""))
        if abstract:
            sections.append(f"**{html.escape(paper.title)}**\n\n{abstract[:1800]}")
    body = "\n\n---\n\n".join(sections) or (
        "No publisher abstracts were available in the current result set."
    )
    return f"> ⚠️ {FALLBACK_SUMMARY_WARNING}\n\n{body}"


@_contextmanager
def workspace_connection(path: str | None = None):
    """Open a transaction-safe SQLite workspace connection."""
    db_path = path or WORKSPACE_DB_PATH
    if db_path != ":memory:":
        _Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = _sqlite3.connect(db_path, timeout=10)
    connection.row_factory = _sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        _initialize_workspace_schema(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_workspace_schema(connection) -> None:
    """Create workspace tables idempotently using one shared connection."""
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workspace_papers (
            paper_key TEXT PRIMARY KEY, title TEXT NOT NULL,
            authors_json TEXT NOT NULL DEFAULT '[]', abstract TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '', published TEXT NOT NULL DEFAULT '',
            doi TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workspace_collections (
            collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE, description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workspace_collection_items (
            collection_id INTEGER NOT NULL, paper_key TEXT NOT NULL,
            added_at TEXT NOT NULL, PRIMARY KEY(collection_id,paper_key),
            FOREIGN KEY(collection_id) REFERENCES workspace_collections(collection_id) ON DELETE CASCADE,
            FOREIGN KEY(paper_key) REFERENCES workspace_papers(paper_key) ON DELETE CASCADE
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workspace_notes (
            note_id INTEGER PRIMARY KEY AUTOINCREMENT, paper_key TEXT,
            title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]', updated_at TEXT NOT NULL,
            FOREIGN KEY(paper_key) REFERENCES workspace_papers(paper_key) ON DELETE SET NULL
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS workspace_searches (
            search_id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
            field_name TEXT NOT NULL DEFAULT 'All fields', result_count INTEGER NOT NULL DEFAULT 0,
            result_keys_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
        )
    """)
    connection.execute("CREATE INDEX IF NOT EXISTS idx_ws_note_paper ON workspace_notes(paper_key)")
    connection.execute("CREATE INDEX IF NOT EXISTS idx_ws_search_date ON workspace_searches(created_at)")


def paper_to_record(paper: Paper) -> dict[str, Any]:
    """Convert a Paper instance to a JSON-compatible dictionary."""
    return {
        "title": normalize_whitespace(paper.title), "authors": list(paper.authors or ()),
        "abstract": str(paper.abstract or ""), "url": str(paper.url or ""),
        "published": str(paper.published or ""), "updated": str(paper.updated or ""),
        "categories": list(paper.categories or ()), "doi": str(paper.doi or ""),
    }


def record_to_paper(record: dict[str, Any]) -> Paper:
    """Reconstruct a Paper from a dictionary while tolerating missing fields."""
    return Paper(
        title=normalize_whitespace(record.get("title", "")) or "Untitled paper",
        authors=tuple(str(x) for x in record.get("authors", []) if x),
        abstract=str(record.get("abstract", "") or ""),
        url=str(record.get("url", "") or ""),
        published=str(record.get("published", "") or ""),
        updated=str(record.get("updated", "") or ""),
        categories=tuple(str(x) for x in record.get("categories", []) if x),
        doi=str(record.get("doi", "") or ""),
    )


def save_workspace_paper(paper: Paper, path: str | None = None) -> str:
    """Save or update a paper locally and return its deterministic workspace key."""
    record = paper_to_record(paper)
    key = stable_cache_key("paper", {
        "doi": record["doi"].lower(), "url": record["url"].lower(),
        "title": record["title"].casefold(),
    })
    metadata = {"updated": record["updated"], "categories": record["categories"]}
    with workspace_connection(path) as connection:
        connection.execute("""
            INSERT INTO workspace_papers
            (paper_key,title,authors_json,abstract,url,published,doi,metadata_json,created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(paper_key) DO UPDATE SET
            title=excluded.title, authors_json=excluded.authors_json,
            abstract=excluded.abstract, url=excluded.url, published=excluded.published,
            doi=excluded.doi, metadata_json=excluded.metadata_json
        """, (
            key, record["title"], json.dumps(record["authors"]), record["abstract"],
            record["url"], record["published"], record["doi"], json.dumps(metadata),
            report_timestamp(),
        ))
    return key


def list_workspace_papers(limit: int = 200, path: str | None = None) -> list[dict[str, Any]]:
    """List saved papers with a bounded maximum number of records."""
    count = bounded_integer(limit, 200, 1, 1000)
    with workspace_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM workspace_papers ORDER BY created_at DESC LIMIT ?", (count,)
        ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["authors"] = json.loads(item.pop("authors_json") or "[]")
        item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        output.append(item)
    return output


def create_workspace_collection(name: str, description: str = "",
                                path: str | None = None) -> int:
    """Create a named paper collection and return its integer ID."""
    clean_name = normalize_whitespace(name)[:100]
    if not clean_name:
        raise ValueError("Collection name cannot be empty.")
    with workspace_connection(path) as connection:
        cursor = connection.execute(
            "INSERT INTO workspace_collections(name,description,created_at) VALUES (?,?,?)",
            (clean_name, normalize_whitespace(description)[:1000], report_timestamp()),
        )
        return int(cursor.lastrowid)


def list_workspace_collections(path: str | None = None) -> list[dict[str, Any]]:
    """Return collection metadata and the number of saved papers in each."""
    with workspace_connection(path) as connection:
        rows = connection.execute("""
            SELECT c.collection_id,c.name,c.description,c.created_at,
                   COUNT(i.paper_key) AS paper_count
            FROM workspace_collections c
            LEFT JOIN workspace_collection_items i ON i.collection_id=c.collection_id
            GROUP BY c.collection_id ORDER BY c.name COLLATE NOCASE
        """).fetchall()
    return [dict(row) for row in rows]


def add_paper_to_collection(collection_id: int, paper: Paper,
                            path: str | None = None) -> bool:
    """Add a saved paper to a collection; return False when already present."""
    paper_key = save_workspace_paper(paper, path)
    with workspace_connection(path) as connection:
        collection = connection.execute(
            "SELECT 1 FROM workspace_collections WHERE collection_id=?", (int(collection_id),)
        ).fetchone()
        if not collection:
            raise ValueError("Collection does not exist.")
        cursor = connection.execute("""
            INSERT OR IGNORE INTO workspace_collection_items(collection_id,paper_key,added_at)
            VALUES (?,?,?)
        """, (int(collection_id), paper_key, report_timestamp()))
        return cursor.rowcount > 0


def save_workspace_note(body: str, *, title: str = "", paper_key: str | None = None,
                        tags: list[str] | None = None, path: str | None = None) -> int:
    """Save a bounded research note, optionally attached to a saved paper."""
    clean_body = str(body or "").strip()
    if not clean_body:
        raise ValueError("Note body cannot be empty.")
    if len(clean_body) > 50000:
        raise ValueError("Note exceeds the 50,000-character limit.")
    clean_tags = list(dict.fromkeys(
        normalize_whitespace(tag)[:40] for tag in (tags or []) if normalize_whitespace(tag)
    ))[:30]
    with workspace_connection(path) as connection:
        cursor = connection.execute("""
            INSERT INTO workspace_notes(paper_key,title,body,tags_json,updated_at)
            VALUES (?,?,?,?,?)
        """, (paper_key, normalize_whitespace(title)[:200], clean_body,
              json.dumps(clean_tags), report_timestamp()))
        return int(cursor.lastrowid)


def list_workspace_notes(limit: int = 200, path: str | None = None) -> list[dict[str, Any]]:
    """List recent workspace notes and decode their tag arrays."""
    with workspace_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM workspace_notes ORDER BY updated_at DESC LIMIT ?",
            (bounded_integer(limit, 200, 1, 1000),),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["tags"] = json.loads(item.pop("tags_json") or "[]")
        result.append(item)
    return result


def record_workspace_search(query: str, field_name: str, papers: list[Paper],
                            path: str | None = None) -> int:
    """Record query metadata and paper keys without persisting provider credentials."""
    clean_query = normalize_whitespace(query)[:MAX_QUERY_CHARS]
    if not clean_query:
        raise ValueError("Search query cannot be empty.")
    keys = [stable_cache_key("paper", paper_to_record(p)) for p in (papers or [])[:MAX_RESULTS]]
    with workspace_connection(path) as connection:
        cursor = connection.execute("""
            INSERT INTO workspace_searches(query,field_name,result_count,result_keys_json,created_at)
            VALUES (?,?,?,?,?)
        """, (clean_query, normalize_whitespace(field_name)[:50], len(keys),
              json.dumps(keys), report_timestamp()))
        return int(cursor.lastrowid)


def list_workspace_searches(limit: int = 50, path: str | None = None) -> list[dict[str, Any]]:
    """List recent workspace searches with bounded result size."""
    with workspace_connection(path) as connection:
        rows = connection.execute(
            "SELECT * FROM workspace_searches ORDER BY created_at DESC LIMIT ?",
            (bounded_integer(limit, 50, 1, 500),),
        ).fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item["result_keys"] = json.loads(item.pop("result_keys_json") or "[]")
        output.append(item)
    return output


def export_workspace_json(path: str | None = None) -> str:
    """Export workspace data to a versioned portable JSON document."""
    with workspace_connection(path) as connection:
        tables = {}
        for table in ("workspace_papers", "workspace_collections",
                      "workspace_collection_items", "workspace_notes", "workspace_searches"):
            tables[table] = [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]
    return robust_json_dumps({
        "schema_version": 1, "exported_at": report_timestamp(), "tables": tables,
    })


def workspace_health(path: str | None = None) -> dict[str, Any]:
    """Check workspace database integrity and return record counts."""
    try:
        with workspace_connection(path) as connection:
            counts = {}
            for table in ("workspace_papers", "workspace_collections",
                          "workspace_collection_items", "workspace_notes", "workspace_searches"):
                counts[table] = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {"ok": integrity == "ok", "integrity": integrity, "counts": counts}
    except Exception as exc:
        log.exception("Workspace health check failed")
        return {"ok": False, "error_type": type(exc).__name__}


def browser_state_script(storage_key: str = "ai-research-scientist-v1") -> str:
    """Build a browser localStorage helper script; UI must explicitly include it."""
    key = json.dumps(str(storage_key)[:100])
    return """
<script>
(() => {
  const KEY = """ + key + """;
  const MAX_CHARS = 250000;
  window.AIRSS = window.AIRSS || {};
  window.AIRSS.save = function(state) {
    try {
      const value = JSON.stringify({version:1, savedAt:Date.now(), state});
      if (value.length > MAX_CHARS) return {ok:false, reason:"state-too-large"};
      localStorage.setItem(KEY, value);
      return {ok:true};
    } catch (_) { return {ok:false, reason:"storage-unavailable"}; }
  };
  window.AIRSS.load = function() {
    try {
      const value = localStorage.getItem(KEY);
      if (!value) return null;
      const parsed = JSON.parse(value);
      return parsed && parsed.version === 1 ? parsed.state : null;
    } catch (_) { return null; }
  };
  window.AIRSS.clear = function() {
    try { localStorage.removeItem(KEY); return true; }
    catch (_) { return false; }
  };
})();
</script>
"""


def normalize_user_workspace_state(value: Any) -> dict[str, Any]:
    """Validate browser-restored state and drop malformed or oversized values."""
    empty = {"query": "", "field": "All fields", "papers": [], "selected": [], "notes": []}
    if not isinstance(value, dict):
        return empty
    field_name = value.get("field", "All fields")
    if field_name not in {"All fields", "Title", "Author", "Abstract"}:
        field_name = "All fields"
    papers = value.get("papers", [])
    selected = value.get("selected", [])
    notes = value.get("notes", [])
    return {
        "query": normalize_whitespace(value.get("query", ""))[:MAX_QUERY_CHARS],
        "field": field_name,
        "papers": [x for x in papers[:MAX_RESULTS] if isinstance(x, dict)] if isinstance(papers, list) else [],
        "selected": [str(x)[:300] for x in selected[:MAX_RESULTS]] if isinstance(selected, list) else [],
        "notes": [x for x in notes[:100] if isinstance(x, dict)] if isinstance(notes, list) else [],
    }


def keyword_frequency(texts: list[str], limit: int = 30) -> list[tuple[str, int]]:
    """Count normalized keywords across a list of texts."""
    counts = Counter()
    for text_value in texts or []:
        counts.update(tokenize_keywords(str(text_value or "")))
    return counts.most_common(bounded_integer(limit, 30, 1, 200))


def cooccurrence_matrix(texts: list[str], top_terms: int = 30) -> dict[str, dict[str, int]]:
    """Create a symmetric keyword co-occurrence matrix for graph displays."""
    documents = [set(tokenize_keywords(t)) for t in (texts or []) if t]
    frequencies = Counter(term for doc in documents for term in doc)
    terms = [term for term, _ in frequencies.most_common(bounded_integer(top_terms, 30, 2, 100))]
    matrix = {term: {other: 0 for other in terms} for term in terms}
    for document in documents:
        present = sorted(document.intersection(terms))
        for index, term in enumerate(present):
            for other in present[index + 1:]:
                matrix[term][other] += 1
                matrix[other][term] += 1
    return matrix


def jaccard_similarity(left: Any, right: Any) -> float:
    """Calculate token-set Jaccard similarity, returning zero for empty inputs."""
    a = set(tokenize_keywords(str(left or "")))
    b = set(tokenize_keywords(str(right or "")))
    union = a | b
    return round(len(a & b) / len(union), 4) if union else 0.0


def paper_similarity(left: Paper, right: Paper) -> dict[str, float]:
    """Compare paper title, abstract, authors, and topics independently."""
    return {
        "title": jaccard_similarity(left.title, right.title),
        "abstract": jaccard_similarity(left.abstract, right.abstract),
        "authors": jaccard_similarity(" ".join(left.authors), " ".join(right.authors)),
        "topics": jaccard_similarity(" ".join(left.categories), " ".join(right.categories)),
    }


def rank_papers_for_query(papers: list[Paper], query: str) -> list[tuple[Paper, float]]:
    """Rank retrieved papers by transparent weighted token overlap."""
    query_tokens = set(tokenize_keywords(query))
    ranked = []
    for paper in papers or []:
        title = set(tokenize_keywords(paper.title))
        abstract = set(tokenize_keywords(paper.abstract))
        authors = set(tokenize_keywords(" ".join(paper.authors)))
        topics = set(tokenize_keywords(" ".join(paper.categories)))
        score = (
            5 * len(query_tokens & title) + 1.5 * len(query_tokens & abstract)
            + 2 * len(query_tokens & topics) + 0.5 * len(query_tokens & authors)
        ) / max(1, len(query_tokens))
        ranked.append((paper, round(score, 4)))
    return sorted(ranked, key=lambda item: (-item[1], item[0].title.casefold()))


def build_research_gap_signals(papers: list[Paper]) -> dict[str, Any]:
    """Describe rare terms and sparse years within a sample, not true novelty."""
    term_counts = Counter()
    year_counts = Counter()
    author_counts = Counter()
    for paper in papers or []:
        term_counts.update(set(tokenize_keywords(paper.title + " " + paper.abstract)))
        if paper.year:
            year_counts[paper.year] += 1
        author_counts.update(set(paper.authors))
    return {
        "rare_terms": sorted(term for term, count in term_counts.items() if count == 1)[:30],
        "sparse_years": sorted(year for year, count in year_counts.items() if count == 1),
        "single_paper_authors": sum(1 for count in author_counts.values() if count == 1),
        "interpretation": "Descriptive signals from retrieved records only; not proof of novelty.",
    }


def estimate_abstract_readability(text: str) -> dict[str, Any]:
    """Compute rough English readability cues; do not treat as validated scoring."""
    value = normalize_whitespace(text)
    words = re.findall(r"\\b[\\w'-]+\\b", value)
    sentences = [s for s in re.split(r"[.!?]+", value) if s.strip()]
    syllables = 0
    for word in words:
        groups = re.findall(r"[aeiouy]+", word.lower())
        syllables += max(1, len(groups) - (1 if word.lower().endswith("e") and len(groups) > 1 else 0))
    wc, sc = len(words), max(1, len(sentences))
    score = 206.835 - 1.015 * (wc / sc) - 84.6 * (syllables / max(1, wc))
    return {
        "word_count": wc, "sentence_count": len(sentences),
        "mean_words_per_sentence": round(wc / sc, 2),
        "heuristic_readability": round(score, 2) if wc else None,
        "caveat": "English-language heuristic; not validated for all scientific writing.",
    }


def validate_doi(value: str) -> str:
    """Normalize a DOI URL or prefix and reject malformed DOI-like strings."""
    candidate = normalize_whitespace(value).lower()
    candidate = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", candidate)
    return candidate if re.fullmatch(r"10\.\d{4,9}/\S+", candidate) else ""


def safe_external_url(value: str) -> str:
    """Accept HTTP(S) URLs only and reject URLs containing embedded credentials."""
    try:
        parsed = urlparse(str(value or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        if parsed.username or parsed.password:
            return ""
        return str(value).strip()
    except Exception:
        return ""


def export_markdown_literature_matrix(papers: list[Paper]) -> str:
    """Export a concise literature matrix that can be edited by a researcher."""
    rows = [
        "# Literature matrix", "",
        f"Generated {report_timestamp()}", "",
        "| Title | Year | Authors | Topics | Abstract |",
        "|---|---:|---|---|---|",
    ]
    for paper in papers or []:
        title = paper.title.replace("|", "\\\\|").replace("\n", " ")
        authors = ", ".join(paper.authors[:5]).replace("|", "\\\\|")
        topics = ", ".join(paper.categories[:5]).replace("|", "\\\\|")
        rows.append(f"| {title} | {paper.year or '—'} | {authors} | {topics} | {'Available' if paper.abstract else 'Missing'} |")
    return "\n".join(rows)


def export_markdown_review(papers: list[Paper], query: str) -> str:
    """Create a source-grounded reading template from available paper metadata."""
    lines = [
        f"# Literature review workspace: {normalize_whitespace(query)}", "",
        f"Generated at {report_timestamp()}.", "",
        "> Structured reading aid only. Verify findings in full papers.", "",
    ]
    for index, paper in enumerate(papers or [], 1):
        lines.extend([
            f"## {index}. {paper.title}",
            f"- Authors: {', '.join(paper.authors) or 'Not supplied'}",
            f"- Publication date: {paper.published or 'Not supplied'}",
            f"- DOI: {paper.doi or 'Not supplied'}",
            f"- Source: {safe_external_url(paper.url) or 'Not supplied'}", "",
            "### Publisher abstract",
            paper.abstract or "_No abstract supplied by the metadata provider._", "",
            "### Critical reading prompts",
            "- What question does the paper investigate?",
            "- Which data, methods, and assumptions support its conclusions?",
            "- What limitations do the authors acknowledge?",
            "- How does it agree or disagree with other included studies?", "",
        ])
    return "\n".join(lines)


def safe_download_name(name: str, extension: str) -> str:
    """Generate a filesystem-safe filename and restrict the extension."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", normalize_whitespace(name)).strip("-_")[:70]
    ext = re.sub(r"[^A-Za-z0-9]", "", extension).lower()[:10]
    return f"{stem or 'research-export'}.{ext or 'txt'}"


# Curated methodology cues. These are advisory reading prompts, never a score
# of paper quality. The catalog is consumed by research_method_profile().
RESEARCH_METHOD_PROFILES = {
    "machine learning": {
        "methods": ["benchmark evaluation", "ablation", "cross-validation", "external validation"],
        "evidence": ["dataset provenance", "baseline comparison", "error analysis", "reproducibility"],
        "cautions": ["data leakage", "distribution shift", "unreported tuning", "benchmark overfitting"],
        "questions": ["Are splits independent?", "Are baselines competitive?", "Are confidence intervals reported?"],
    },
    "deep learning": {
        "methods": ["multi-seed evaluation", "architecture comparison", "ablation", "external validation"],
        "evidence": ["training details", "compute budget", "dataset provenance", "failure examples"],
        "cautions": ["seed sensitivity", "hidden preprocessing", "compute inequality", "baseline tuning"],
        "questions": ["Are seeds and variance reported?", "Can the training procedure be reproduced?"],
    },
    "natural language processing": {
        "methods": ["held-out evaluation", "human annotation", "error analysis", "cross-domain evaluation"],
        "evidence": ["annotation guidelines", "inter-annotator agreement", "dataset card", "model card"],
        "cautions": ["benchmark contamination", "language imbalance", "prompt sensitivity", "annotation bias"],
        "questions": ["Which languages are represented?", "Is contamination assessed?"],
    },
    "computer vision": {
        "methods": ["controlled benchmark", "cross-dataset evaluation", "ablation", "robustness testing"],
        "evidence": ["dataset splits", "image acquisition", "calibration", "failure analysis"],
        "cautions": ["near-duplicate leakage", "domain shift", "class imbalance", "spurious backgrounds"],
        "questions": ["Does performance transfer to new sources?", "Are per-class results reported?"],
    },
    "medical imaging": {
        "methods": ["patient-level split", "external-site validation", "reader study", "prospective evaluation"],
        "evidence": ["cohort definition", "reference standard", "sensitivity and specificity", "calibration"],
        "cautions": ["patient leakage", "spectrum bias", "site confounding", "workflow mismatch"],
        "questions": ["Are splits patient-level?", "Was external clinical validation performed?"],
    },
    "clinical medicine": {
        "methods": ["randomized trial", "cohort study", "case-control study", "systematic review"],
        "evidence": ["protocol registration", "participant flow", "effect sizes", "adverse events"],
        "cautions": ["confounding", "selection bias", "missing outcomes", "multiple testing"],
        "questions": ["Was the protocol prespecified?", "Are absolute effects and harms reported?"],
    },
    "public health": {
        "methods": ["population survey", "natural experiment", "cohort study", "interrupted time series"],
        "evidence": ["sampling frame", "response rate", "confounder handling", "sensitivity analysis"],
        "cautions": ["ecological fallacy", "nonresponse bias", "measurement error", "temporal confounding"],
        "questions": ["Who is missing from the sample?", "Could another event explain the trend?"],
    },
    "climate science": {
        "methods": ["observation analysis", "ensemble modeling", "attribution analysis", "scenario comparison"],
        "evidence": ["measurement uncertainty", "model validation", "scenario assumptions", "sensitivity analysis"],
        "cautions": ["internal variability", "scenario dependence", "spatial resolution", "baseline selection"],
        "questions": ["Which scenario is used?", "How is uncertainty propagated?"],
    },
    "renewable energy": {
        "methods": ["techno-economic analysis", "life-cycle assessment", "field measurement", "system simulation"],
        "evidence": ["capacity factor", "system boundary", "cost assumptions", "lifetime estimates"],
        "cautions": ["optimistic cost assumptions", "grid constraints", "lifecycle omissions", "site dependence"],
        "questions": ["Are lifecycle impacts included?", "Are costs sensitivity-tested?"],
    },
    "cybersecurity": {
        "methods": ["threat modeling", "controlled attack evaluation", "red-team assessment", "formal verification"],
        "evidence": ["attacker model", "reproducible setup", "false-positive rate", "responsible disclosure"],
        "cautions": ["unrealistic threat model", "dataset obsolescence", "adaptive adversaries", "deployment mismatch"],
        "questions": ["What attacker capabilities are assumed?", "Are false positives evaluated?"],
    },
    "robotics": {
        "methods": ["simulation benchmark", "hardware trial", "ablation study", "real-world transfer"],
        "evidence": ["robot configuration", "environment details", "success criteria", "safety protocol"],
        "cautions": ["sim-to-real gap", "small trial counts", "environment overfitting", "hardware confounding"],
        "questions": ["Does it work outside simulation?", "Are failures and safety events reported?"],
    },
    "bioinformatics": {
        "methods": ["cross-validation", "external cohort validation", "benchmark comparison", "functional validation"],
        "evidence": ["data accession", "preprocessing pipeline", "multiple-testing correction", "code availability"],
        "cautions": ["sample leakage", "batch effects", "population stratification", "label leakage"],
        "questions": ["Are cohorts independent?", "Are batch effects and multiplicity handled?"],
    },
    "education research": {
        "methods": ["randomized intervention", "quasi-experiment", "longitudinal study", "mixed methods"],
        "evidence": ["intervention fidelity", "validated instruments", "attrition", "context description"],
        "cautions": ["teacher effects", "selection bias", "short follow-up", "non-independent observations"],
        "questions": ["Was implementation consistent?", "Are long-term outcomes measured?"],
    },
    "economics": {
        "methods": ["natural experiment", "instrumental variables", "difference-in-differences", "structural modeling"],
        "evidence": ["identification assumptions", "robustness checks", "data provenance", "effect sizes"],
        "cautions": ["violated exclusion restrictions", "parallel-trends failure", "endogeneity", "specification search"],
        "questions": ["Which assumptions identify the effect?", "Are placebo tests convincing?"],
    },
    "psychology": {
        "methods": ["randomized experiment", "longitudinal study", "survey study", "replication"],
        "evidence": ["preregistration", "validated measures", "sample-size rationale", "effect-size intervals"],
        "cautions": ["small samples", "researcher degrees of freedom", "measurement validity", "sampling bias"],
        "questions": ["Was the study preregistered?", "Do findings replicate in different populations?"],
    },
    "social science": {
        "methods": ["survey research", "ethnography", "case study", "mixed methods"],
        "evidence": ["sampling rationale", "coding framework", "reflexivity", "triangulation"],
        "cautions": ["researcher positionality", "nonresponse", "context transfer", "construct validity"],
        "questions": ["How were participants selected?", "Are interpretations supported by the data?"],
    },
    "materials science": {
        "methods": ["controlled synthesis", "characterization", "simulation", "accelerated aging"],
        "evidence": ["sample preparation", "instrument calibration", "replicate batches", "uncertainty"],
        "cautions": ["batch variation", "instrument drift", "small sample sizes", "unrealistic conditions"],
        "questions": ["Are results replicated across batches?", "Are instruments calibrated?"],
    },
    "physics": {
        "methods": ["experimental measurement", "numerical simulation", "analytical derivation", "replication"],
        "evidence": ["uncertainty budget", "calibration", "model assumptions", "independent replication"],
        "cautions": ["systematic error", "selection effects", "model dependence", "unreported exclusions"],
        "questions": ["Are systematic uncertainties quantified?", "Can the result be independently reproduced?"],
    },
    "chemistry": {
        "methods": ["controlled experiment", "spectroscopic characterization", "kinetic modeling", "replication"],
        "evidence": ["reagent purity", "instrument settings", "replicate measurements", "data availability"],
        "cautions": ["contamination", "selective reporting", "scale-up effects", "instrument calibration"],
        "questions": ["Are raw spectra available?", "Are yields and uncertainty reported?"],
    },
    "agriculture": {
        "methods": ["field trial", "randomized block design", "longitudinal monitoring", "remote sensing"],
        "evidence": ["site characteristics", "seasonal conditions", "soil properties", "site replication"],
        "cautions": ["weather dependence", "soil heterogeneity", "short follow-up", "external validity"],
        "questions": ["Are results replicated across seasons?", "Are environmental conditions documented?"],
    },
    "transportation": {
        "methods": ["traffic simulation", "natural experiment", "before-after study", "field deployment"],
        "evidence": ["network conditions", "exposure measurement", "safety outcomes", "demand assumptions"],
        "cautions": ["induced demand", "seasonality", "spillover effects", "simulation calibration"],
        "questions": ["Are spillovers measured?", "Is the model calibrated against real observations?"],
    },
    "human-computer interaction": {
        "methods": ["usability study", "controlled user experiment", "field deployment", "think-aloud protocol"],
        "evidence": ["participant demographics", "task definitions", "completion time", "qualitative coding"],
        "cautions": ["convenience sampling", "novelty effects", "task artificiality", "small user groups"],
        "questions": ["Do tasks represent real use?", "Are accessibility needs considered?"],
    },
    "information retrieval": {
        "methods": ["test collection evaluation", "online experiment", "ranking comparison", "user study"],
        "evidence": ["query set", "relevance judgments", "NDCG or MAP", "latency measurement"],
        "cautions": ["query leakage", "judgment bias", "offline-online mismatch", "incomplete pooling"],
        "questions": ["Are judgments sufficiently pooled?", "Are latency and quality both evaluated?"],
    },
    "data mining": {
        "methods": ["benchmark comparison", "cross-validation", "ablation study", "subgroup analysis"],
        "evidence": ["data cleaning rules", "feature definitions", "baseline metrics", "reproducible code"],
        "cautions": ["target leakage", "selection bias", "multiple comparisons", "dataset shift"],
        "questions": ["Was preprocessing fit only on training data?", "Are subgroup failures visible?"],
    },
    "statistics": {
        "methods": ["simulation study", "asymptotic analysis", "bootstrap analysis", "estimator comparison"],
        "evidence": ["assumption statements", "coverage rates", "bias and variance", "code and seeds"],
        "cautions": ["miscalibrated intervals", "asymptotic misuse", "simulation design bias", "model misspecification"],
        "questions": ["Are assumptions explicit?", "Does finite-sample behavior match theory?"],
    },
    "operations research": {
        "methods": ["optimization benchmark", "simulation", "sensitivity analysis", "case study"],
        "evidence": ["objective function", "constraints", "solver configuration", "optimality gap"],
        "cautions": ["unrealistic constraints", "solver tuning bias", "scale mismatch", "unreported infeasibility"],
        "questions": ["Are constraints grounded in practice?", "Are runtime and optimality gaps reported?"],
    },
    "software engineering": {
        "methods": ["controlled experiment", "repository mining", "case study", "replication package"],
        "evidence": ["project sampling", "operational definitions", "statistical analysis", "artifact availability"],
        "cautions": ["repository selection bias", "confounding", "metric validity", "publication bias"],
        "questions": ["Are projects representative?", "Are metrics valid proxies for the claimed construct?"],
    },
    "human genetics": {
        "methods": ["association study", "replication cohort", "fine mapping", "functional follow-up"],
        "evidence": ["ancestry composition", "multiple-testing correction", "variant QC", "replication"],
        "cautions": ["population stratification", "winner's curse", "ancestry imbalance", "phenotype misclassification"],
        "questions": ["Are findings replicated across ancestries?", "Are association and causation distinguished?"],
    },
    "epidemiology": {
        "methods": ["prospective cohort", "case-control", "surveillance analysis", "target-trial emulation"],
        "evidence": ["case definition", "exposure measurement", "confounder adjustment", "sensitivity analysis"],
        "cautions": ["immortal-time bias", "residual confounding", "selection bias", "misclassification"],
        "questions": ["Is time zero defined consistently?", "Are unmeasured confounders discussed?"],
    },
    "linguistics": {
        "methods": ["corpus analysis", "elicitation study", "fieldwork", "experimental linguistics"],
        "evidence": ["corpus provenance", "speaker demographics", "annotation scheme", "inter-rater agreement"],
        "cautions": ["corpus imbalance", "elicitation effects", "language endangerment", "coding subjectivity"],
        "questions": ["Who is represented in the corpus?", "Can coding be independently reproduced?"],
    },
    "history": {
        "methods": ["archival analysis", "source criticism", "comparative case study", "oral history"],
        "evidence": ["primary-source provenance", "contextual triangulation", "chronology", "source limitations"],
        "cautions": ["archival survivorship", "translation bias", "retrospective memory", "presentism"],
        "questions": ["Who produced the sources and why?", "Which voices are absent from the archive?"],
    },
    "law": {
        "methods": ["doctrinal analysis", "comparative legal analysis", "empirical legal study", "case analysis"],
        "evidence": ["jurisdiction", "primary authorities", "procedural posture", "date of law"],
        "cautions": ["jurisdiction mismatch", "outdated authorities", "case selection", "normative-descriptive conflation"],
        "questions": ["Which jurisdiction and date apply?", "Are primary legal authorities cited?"],
    },
    "finance": {
        "methods": ["event study", "portfolio backtest", "panel regression", "risk modeling"],
        "evidence": ["transaction costs", "out-of-sample test", "risk adjustment", "data timestamps"],
        "cautions": ["look-ahead bias", "survivorship bias", "multiple testing", "regime changes"],
        "questions": ["Are transaction costs included?", "Was the strategy tested out of sample?"],
    },
    "environmental science": {
        "methods": ["field monitoring", "life-cycle assessment", "remote sensing", "ecological experiment"],
        "evidence": ["sampling design", "seasonality", "measurement uncertainty", "site replication"],
        "cautions": ["temporal autocorrelation", "site specificity", "detection limits", "confounding stressors"],
        "questions": ["Is seasonal variability covered?", "Are detection limits reported?"],
    },
    "systematic reviews": {
        "methods": ["protocol-driven search", "dual screening", "risk-of-bias assessment", "evidence synthesis"],
        "evidence": ["search strings", "eligibility criteria", "flow diagram", "certainty assessment"],
        "cautions": ["publication bias", "database coverage", "selective outcomes", "heterogeneity"],
        "questions": ["Can the search be reproduced?", "Are exclusions and reasons documented?"],
    },
    "meta-analysis": {
        "methods": ["random-effects synthesis", "sensitivity analysis", "subgroup analysis", "small-study assessment"],
        "evidence": ["effect sizes", "heterogeneity", "study weights", "publication bias assessment"],
        "cautions": ["dependent effects", "selective reporting", "small-study effects", "incompatible outcomes"],
        "questions": ["Are effects independent?", "Is heterogeneity interpreted, not just reported?"],
    },
    "qualitative research": {
        "methods": ["thematic analysis", "grounded theory", "phenomenology", "ethnography"],
        "evidence": ["sampling rationale", "coding process", "reflexivity", "negative cases"],
        "cautions": ["decontextualized quotes", "overclaiming saturation", "researcher influence", "thin description"],
        "questions": ["How were interpretations developed?", "Are contradictory cases addressed?"],
    },
    "mixed methods": {
        "methods": ["sequential explanatory", "sequential exploratory", "convergent design", "embedded design"],
        "evidence": ["integration rationale", "sampling alignment", "joint displays", "divergent findings"],
        "cautions": ["unintegrated strands", "unequal weighting", "integration afterthought", "incompatible samples"],
        "questions": ["Where are strands integrated?", "Are contradictions explained?"],
    },
    "causal inference": {
        "methods": ["randomization", "instrumental variables", "regression discontinuity", "difference-in-differences"],
        "evidence": ["causal graph", "identification assumptions", "placebo tests", "sensitivity analysis"],
        "cautions": ["post-treatment adjustment", "weak instruments", "threshold manipulation", "parallel-trend violations"],
        "questions": ["Which assumptions are testable?", "How sensitive are estimates to violations?"],
    },
    "reproducibility": {
        "methods": ["independent replication", "multiverse analysis", "registered report", "artifact evaluation"],
        "evidence": ["versioned data", "environment lockfile", "analysis scripts", "preregistration"],
        "cautions": ["undocumented steps", "data restrictions", "version drift", "selective success reporting"],
        "questions": ["Can an independent team reproduce the result?", "Are deviations disclosed?"],
    },
    "scientometrics": {
        "methods": ["bibliometric analysis", "citation network analysis", "co-word analysis", "topic modeling"],
        "evidence": ["database coverage", "deduplication rules", "field normalization", "time window"],
        "cautions": ["citation lag", "database bias", "name ambiguity", "metric gaming"],
        "questions": ["Are counts normalized by field and year?", "How are missing and duplicate records handled?"],
    },
    "research evaluation": {
        "methods": ["peer review analysis", "portfolio analysis", "qualitative assessment", "indicator validation"],
        "evidence": ["evaluation criteria", "stakeholder goals", "uncertainty", "contextual interpretation"],
        "cautions": ["Goodhart effects", "metric monoculture", "discipline bias", "short-termism"],
        "questions": ["Does the indicator reflect the intended construct?", "Could metrics change behavior undesirably?"],
    },
    "large language models": {
        "methods": ["benchmark evaluation", "human preference study", "red teaming", "long-context evaluation"],
        "evidence": ["prompt protocol", "model version", "sampling parameters", "contamination checks"],
        "cautions": ["benchmark contamination", "prompt sensitivity", "judge bias", "nondeterminism"],
        "questions": ["Is model version pinned?", "Are human evaluations blinded and diverse?"],
    },
    "retrieval augmented generation": {
        "methods": ["retrieval ablation", "faithfulness evaluation", "citation verification", "latency-cost analysis"],
        "evidence": ["retrieval corpus", "chunking configuration", "retrieval metrics", "groundedness judgments"],
        "cautions": ["retrieval misses", "citation mismatch", "corpus leakage", "answer overconfidence"],
        "questions": ["Are citations entailed by retrieved passages?", "Are retrieval and generation errors separated?"],
    },
    "vector databases": {
        "methods": ["recall-latency benchmark", "index parameter sweep", "filter evaluation", "load testing"],
        "evidence": ["embedding model", "corpus size", "hardware", "index configuration"],
        "cautions": ["unfair hardware comparison", "stale embeddings", "distribution shift", "memory overhead"],
        "questions": ["Are recall and latency both measured?", "Are updates and deletes benchmarked?"],
    },
    "open science": {
        "methods": ["preregistration audit", "data availability audit", "code reuse study", "replication"],
        "evidence": ["persistent identifiers", "license", "data dictionary", "materials archive"],
        "cautions": ["privacy risks", "inaccessible repositories", "ambiguous licenses", "performative openness"],
        "questions": ["Are artifacts findable and usable?", "Does sharing protect participant privacy?"],
    },
    "research ethics": {
        "methods": ["ethics review analysis", "participant interview", "case study", "policy evaluation"],
        "evidence": ["consent process", "risk mitigation", "participant voice", "data governance"],
        "cautions": ["coercion", "vulnerability labeling", "privacy risks", "institutional conflicts"],
        "questions": ["Can participants withdraw meaningfully?", "Are benefits and risks fairly distributed?"],
    },
    "AI research tools": {
        "methods": ["task benchmark", "user study", "error audit", "workflow evaluation"],
        "evidence": ["source traceability", "hallucination rate", "latency", "user corrections"],
        "cautions": ["fabricated citations", "automation bias", "provider outages", "privacy exposure"],
        "questions": ["Can generated claims be traced to sources?", "Does the tool fail safely when providers fail?"],
    },
    "AI summarization": {
        "methods": ["factuality evaluation", "human preference study", "coverage analysis", "robustness test"],
        "evidence": ["source text", "claim-level support", "omission rate", "summary length"],
        "cautions": ["hallucination", "important omission", "overcompression", "prompt sensitivity"],
        "questions": ["Can each claim be traced to source text?", "Are limitations and conflicting results retained?"],
    },
    "web scraping": {
        "methods": ["coverage audit", "data validation", "crawl reproducibility", "change monitoring"],
        "evidence": ["source policy", "crawl timestamps", "parser tests", "rate-limit behavior"],
        "cautions": ["terms violations", "robots restrictions", "site changes", "personal data exposure"],
        "questions": ["Is there an official API?", "Does collection respect restrictions and minimize data?"],
    },
    "API reliability": {
        "methods": ["fault injection", "load test", "contract test", "latency analysis"],
        "evidence": ["error rates", "retry policy", "timeout", "idempotency"],
        "cautions": ["retry storms", "duplicate writes", "unbounded latency", "dependency failure"],
        "questions": ["Are retries bounded and jittered?", "Are non-idempotent operations protected?"],
    },
    "edge caching": {
        "methods": ["cache hit analysis", "stale-data evaluation", "load test", "invalidation study"],
        "evidence": ["TTL", "cache key", "purge mechanism", "origin load"],
        "cautions": ["personal data caching", "cache poisoning", "stale content", "key collisions"],
        "questions": ["Are user-specific responses isolated?", "Are invalidation and stale-data behavior tested?"],
    },
    "research platform reliability": {
        "methods": ["fault injection", "API contract testing", "load testing", "recovery drill"],
        "evidence": ["availability", "error rate", "retry policy", "fallback behavior"],
        "cautions": ["retry storms", "silent failure", "provider dependence", "state loss"],
        "questions": ["Does failure preserve user work?", "Are retries bounded, observable, and safe?"],
    },
}


def research_method_profile(query: str) -> dict[str, Any]:
    """Find the closest methodology profile using normalized token overlap."""
    tokens = set(tokenize_keywords(query))
    best_key, best_score = "", 0.0
    for key in RESEARCH_METHOD_PROFILES:
        key_tokens = set(tokenize_keywords(key))
        score = len(tokens & key_tokens) / max(1, len(tokens | key_tokens))
        if score > best_score:
            best_key, best_score = key, score
    if not best_key or best_score < 0.12:
        return {
            "matched_topic": None, "confidence": 0.0, "methods": [],
            "evidence": [], "cautions": [],
            "questions": ["What is the study design?", "What evidence supports the claim?",
                          "What are the main limitations?"],
        }
    result = dict(RESEARCH_METHOD_PROFILES[best_key])
    result.update({"matched_topic": best_key, "confidence": round(best_score, 3)})
    return result


def methodology_review_checklist(query: str) -> list[str]:
    """Build a practical critical-reading checklist for a research topic."""
    profile = research_method_profile(query)
    checklist = [
        "Identify the exact research question and intended population.",
        "Separate reported results from your own interpretation.",
        "Check whether the method matches the research question.",
        "Inspect sample selection, data provenance, and missing observations.",
        "Check whether baselines and controls are appropriate.",
        "Look for uncertainty intervals, effect sizes, and practical significance.",
        "Inspect funding, conflicts of interest, and ethical safeguards.",
        "Open the source and verify the paper version before relying on claims.",
    ]
    checklist.extend(f"Method cue: {item}" for item in profile.get("methods", [])[:4])
    checklist.extend(f"Potential caveat: {item}" for item in profile.get("cautions", [])[:4])
    checklist.extend(f"Critical question: {item}" for item in profile.get("questions", [])[:4])
    return list(dict.fromkeys(checklist))


def build_research_brief(papers: list[Paper], query: str) -> dict[str, Any]:
    """Combine result metadata, critical-reading prompts, and a literature matrix."""
    records = papers or []
    return {
        "title": f"Research brief: {normalize_whitespace(query)[:180]}",
        "generated_at": report_timestamp(),
        "query": normalize_whitespace(query)[:MAX_QUERY_CHARS],
        "paper_count": len(records),
        "methodology_profile": research_method_profile(query),
        "review_checklist": methodology_review_checklist(query),
        "landscape": result_set_summary(records),
        "collection_health": collection_health(records),
        "literature_matrix_markdown": export_markdown_literature_matrix(records),
        "caveat": "This brief uses retrieved metadata and abstracts only; read full papers.",
    }


def feature_status_report() -> dict[str, Any]:
    """Return privacy-safe feature status without revealing environment secrets."""
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "python": __import__("sys").version.split()[0],
        "gradio_available": gr is not None,
        "requests_available": requests is not None,
        "openai_package_available": OpenAI is not None,
        "openai_configured": bool(OPENAI_API_KEY and OpenAI is not None),
        "openalex_key_configured": bool(OPENALEX_API_KEY),
        "database_url_configured": bool(DATABASE_URL),
        "cache_entries": len(_MEMORY_CACHE),
        "methodology_profile_count": len(RESEARCH_METHOD_PROFILES),
        "secrets_disclosed": False,
    }


def self_test_research_extensions() -> dict[str, Any]:
    """Run deterministic smoke checks for core helper functions."""
    checks = {}
    checks["stable_cache_key"] = stable_cache_key("x", {"b": 2, "a": 1}) == stable_cache_key("x", {"a": 1, "b": 2})
    test_key = stable_cache_key("smoke", "fixed")
    cache_set(test_key, "ok", 10)
    checks["cache_round_trip"] = cache_get(test_key) == "ok"
    checks["doi_validation"] = validate_doi("https://doi.org/10.1000/example") == "10.1000/example"
    checks["url_validation"] = safe_external_url("https://example.org/paper") == "https://example.org/paper"
    checks["url_rejection"] = safe_external_url("javascript:alert(1)") == ""
    checks["similarity_identity"] = jaccard_similarity("machine learning", "machine learning") == 1.0
    checks["method_profile"] = bool(research_method_profile("machine learning").get("methods"))
    checks["state_validation"] = normalize_user_workspace_state({"field": "bad"})["field"] == "All fields"
    checks["safe_filename"] = safe_download_name("Research export", "csv").endswith(".csv")
    checks["retry_success"] = retry_with_exponential_backoff(lambda: "ok", attempts=1) == "ok"
    return {"ok": all(checks.values()), "checks": checks,
            "passed": sum(checks.values()), "total": len(checks)}


def list_methodology_topics(prefix: str = "", limit: int = 200) -> list[str]:
    """List methodology topics, optionally filtered by case-insensitive prefix."""
    cleaned = normalize_whitespace(prefix).casefold()
    topics = sorted(RESEARCH_METHOD_PROFILES)
    return [topic for topic in topics if topic.startswith(cleaned)][:bounded_integer(limit, 200, 1, 1000)]


def search_methodology_topics(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Rank methodology profiles by overlap with query terms and method vocabulary."""
    tokens = set(tokenize_keywords(query))
    if not tokens:
        return []
    matches = []
    for topic, profile in RESEARCH_METHOD_PROFILES.items():
        profile_tokens = set(tokenize_keywords(topic + " " + " ".join(profile["methods"])))
        score = len(tokens & profile_tokens) / max(1, len(tokens | profile_tokens))
        if score:
            matches.append({"topic": topic, "score": round(score, 4),
                            "methods": profile["methods"][:3],
                            "cautions": profile["cautions"][:3]})
    matches.sort(key=lambda item: (-item["score"], item["topic"]))
    return matches[:bounded_integer(limit, 20, 1, 100)]


def profile_coverage_report() -> dict[str, Any]:
    """Report the size of the curated methodology knowledge base."""
    return {
        "profile_count": len(RESEARCH_METHOD_PROFILES),
        "method_count": sum(len(v["methods"]) for v in RESEARCH_METHOD_PROFILES.values()),
        "caution_count": sum(len(v["cautions"]) for v in RESEARCH_METHOD_PROFILES.values()),
        "question_count": sum(len(v["questions"]) for v in RESEARCH_METHOD_PROFILES.values()),
        "notice": "Profiles are critical-reading aids, not formal quality ratings.",
    }


# Extended, data-driven research domain checklist catalog.
# Entries are heuristics to help review papers, not validated quality scores.
EXTENDED_METHOD_PROFILES = {
    'artificial intelligence': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'machine learning': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'deep learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'natural language processing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'computer vision': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'robotics': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'reinforcement learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'federated learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'continual learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'transfer learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'self-supervised learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'multimodal learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'graph neural networks': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'knowledge graphs': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'information retrieval': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'semantic search': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'recommendation systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'data mining': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'data science': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'statistics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Bayesian inference': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'causal inference': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'time series forecasting': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'optimization': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'operations research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'simulation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'complex systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'network science': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientometrics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'bibliometrics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'citation analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research evaluation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'open science': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'reproducibility': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research ethics': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'clinical trials': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'epidemiology': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'public health': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'health informatics': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'medical imaging': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'radiology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'pathology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'oncology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'cardiology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'infectious disease': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'vaccinology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'mental health': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'psychology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'neuroscience': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'cognitive science': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'education research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'online learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'learning analytics': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'assessment': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'teacher education': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'higher education': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'economics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'development economics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'labor economics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'macroeconomics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'finance': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'financial risk': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'insurance': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'business': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'marketing': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'entrepreneurship': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'innovation studies': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'public policy': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'political science': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'international relations': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'law': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'privacy law': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'intellectual property': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'cybersecurity': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'software engineering': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'software testing': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'API design': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'cloud computing': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'distributed systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'database systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'high performance computing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'parallel computing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'computer networks': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'wireless communications': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    '5G networks': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    '6G networks': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'internet of things': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'edge computing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'edge AI': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'embedded systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'control systems': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'electrical engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'mechanical engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'civil engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'structural engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'geotechnical engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'chemical engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'materials science': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'nanotechnology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'battery research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'hydrogen energy': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'nuclear energy': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'renewable energy': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'energy systems': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'smart grids': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'climate science': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'climate modeling': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'climate adaptation': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'climate mitigation': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'environmental science': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'environmental justice': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'air quality': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'water resources': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'water treatment': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'waste management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'life-cycle assessment': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'circular economy': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'sustainability': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'ecology': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'conservation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'marine biology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'oceanography': {
        'methods': ['field measurement', 'scenario analysis', 'model validation', 'sensitivity analysis'],
        'evidence': ['sampling design', 'uncertainty', 'system boundary', 'temporal coverage'],
        'cautions': ['site dependence', 'nonstationarity', 'boundary omissions', 'measurement error'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'meteorology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'hydrology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'geology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'geophysics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'astronomy': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'astrophysics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'particle physics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'quantum computing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'quantum information': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'mathematical modeling': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'applied mathematics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'numerical analysis': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'computational fluid dynamics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'finite element analysis': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'manufacturing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'additive manufacturing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'reliability engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'safety engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'quality engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'supply chain': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'logistics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'transportation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'urban planning': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'urban economics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'geospatial analysis': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'remote sensing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'geographic information science': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'architecture': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'design research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'human-computer interaction': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'user experience': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'accessibility': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'digital inclusion': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'digital government': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'digital humanities': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'history': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'linguistics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'literary studies': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'translation studies': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'communication': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'journalism': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'media studies': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'misinformation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'science communication': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'science policy': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research funding': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'peer review': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic publishing': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'open access': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific software': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research software engineering': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'data visualization': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'visual analytics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'technical documentation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research project management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'collaborative research': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'interdisciplinary research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research training': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'graduate education': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic careers': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research diversity': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic metrics': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'publication bias': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'meta-research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific misconduct': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'retractions': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific databases': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic search': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research discovery': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research recommendation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'literature review automation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific question answering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI summarization': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'document summarization': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'citation generation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'bibliography management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research comparison': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research landscape analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research gap analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research novelty assessment': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research prioritization': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research agenda setting': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'evidence gap maps': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research question design': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'hypothesis generation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'experimental design': {
        'methods': ['cohort or trial design', 'external validation', 'risk-of-bias assessment', 'patient-centered evaluation'],
        'evidence': ['population definition', 'outcome validity', 'follow-up', 'harms'],
        'cautions': ['confounding', 'selection bias', 'missing outcomes', 'generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'statistical power': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'multiple testing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'effect sizes': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'missing data': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'measurement error': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'psychometrics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'survey methodology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'qualitative research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'mixed methods': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'ethnography': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'interview research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'case study research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Bayesian statistics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'causal machine learning': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'forecasting': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'network analysis': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'topic modeling': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'text mining': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'sentiment analysis': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'social media analysis': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'bot detection': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'content moderation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'online safety': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'digital wellbeing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'media psychology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'consumer privacy': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'dark patterns': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'algorithmic transparency': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'algorithmic accountability': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'algorithmic auditing': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'dataset documentation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'data-centric AI': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'synthetic data': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'data augmentation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'weak supervision': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'active learning': {
        'methods': ['learning outcome evaluation', 'comparison design', 'longitudinal follow-up', 'mixed methods'],
        'evidence': ['learner population', 'assessment validity', 'implementation fidelity', 'retention'],
        'cautions': ['selection bias', 'engagement as proxy', 'short follow-up', 'access inequality'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'human annotation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'crowdsourcing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research automation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'web scraping': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'web data extraction': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'API reliability': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'edge caching': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'distributed caching': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'database performance': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'database migration': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'backup and recovery': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'application security': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'privacy-preserving analytics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'user state persistence': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'frontend performance': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'frontend state management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'accessibility engineering': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'internationalization': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'localization': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific writing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research communication': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'visual abstracts': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific posters': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic presentations': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research workflow automation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic workflow design': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research workspace design': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific dashboard design': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data visualization': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific decision support': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research ethics review': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'informed consent': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'participant recruitment': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research participant privacy': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data sharing': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'open research infrastructure': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'persistent identifiers': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scholarly metadata': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic identifiers': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'author disambiguation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'institutional research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research funding analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'grant evaluation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research infrastructure sustainability': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI research tools': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data caching': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research result ranking': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research search evaluation': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research query expansion': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research terminology': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'multilingual research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'low-resource languages': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'translation technology': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific terminology extraction': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scholarly entity resolution': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'preprint studies': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'peer review transparency': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'publication ethics': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific peer review bias': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research assessment reform': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'responsible metrics': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'open peer review': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'pre-registration': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'registered reports': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data repositories': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scholarly communication': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic libraries': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research productivity': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic writing tools': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'reference managers': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research collaboration ethics': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research mentoring': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific careers': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research workforce diversity': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research well-being': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic labor': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research funding equity': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'institutional research strategy': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research risk management': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific infrastructure': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data infrastructure': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific cloud infrastructure': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research compute': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'green computing': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI energy use': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI infrastructure': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI model serving': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'LLM reliability': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'LLM guardrails': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI hallucination detection': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'retrieval evaluation': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'vector search': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'semantic similarity': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'document embeddings': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'vector database operations': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'semantic caching': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'API rate limiting': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'exponential backoff': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'web proxy reliability': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'web scraping ethics': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'crawler scheduling': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'content extraction': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'academic content extraction': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research API fallback': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'multi-provider research search': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'OpenAlex': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Crossref': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Semantic Scholar': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'PubMed': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'arXiv': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Europe PMC': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research data commons': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research graph analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'co-citation analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'bibliographic coupling': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'co-word analysis': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'topic evolution': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'scientific collaboration': {
        'methods': ['comparative analysis', 'identification strategy', 'robustness checks', 'stakeholder analysis'],
        'evidence': ['sample definition', 'assumptions', 'effect sizes', 'context'],
        'cautions': ['confounding', 'selection effects', 'measurement validity', 'external validity'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research diversity indices': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'Shannon entropy': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'topic diversity': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research interdisciplinarity': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research impact measurement': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research trend forecasting': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'emerging technology research': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research opportunity analysis': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research risk assessment': {
        'methods': ['reproducible search', 'record deduplication', 'source verification', 'coverage analysis'],
        'evidence': ['search strategy', 'source provenance', 'metadata completeness', 'date coverage'],
        'cautions': ['database bias', 'indexing lag', 'duplicate records', 'citation overinterpretation'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research ethics in AI': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI research reproducibility': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI benchmark design': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'benchmark contamination': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'model calibration': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'uncertainty estimation': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'out-of-distribution detection': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'adversarial machine learning': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI red teaming': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI incident reporting': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI monitoring': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI lifecycle management': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'model governance': {
        'methods': ['clear study design', 'independent validation', 'sensitivity analysis', 'transparent reporting'],
        'evidence': ['sampling rationale', 'measurement validity', 'uncertainty', 'replication materials'],
        'cautions': ['selection bias', 'measurement error', 'model assumptions', 'limited generalizability'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI procurement': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI regulation': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI standards': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI assurance': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI safety cases': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI system validation': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI system verification': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI reliability engineering': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI user experience': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI explainability interfaces': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI literacy': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI adoption': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI labor impact': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI education policy': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI healthcare policy': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI environmental impact': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI and scientific discovery': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for materials discovery': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for drug discovery': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for climate science': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for agriculture': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for education': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for accessibility': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for public services': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for law': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for finance': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for cybersecurity': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for robotics': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'AI for scientific literature': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform usability': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform performance': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform security': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform observability': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform deployment': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform maintenance': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform testing': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform accessibility': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform state': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform caching': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform fallbacks': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform rate limits': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform API design': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform data integrity': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform privacy': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform error handling': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform input validation': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform file export': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform bibliography': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform analytics': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform AI synthesis': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform search quality': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform user control': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform resilience': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform maintainability': {
        'methods': ['benchmark evaluation', 'external validation', 'ablation study', 'failure analysis'],
        'evidence': ['data provenance', 'baseline quality', 'error distribution', 'reproducibility'],
        'cautions': ['data leakage', 'distribution shift', 'metric gaming', 'overclaiming'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform scalability': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform deployment safety': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform user trust': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform documentation': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
    'research platform roadmap': {
        'methods': ['contract testing', 'load testing', 'failure injection', 'security review'],
        'evidence': ['versioned configuration', 'latency and errors', 'recovery behavior', 'test artifacts'],
        'cautions': ['environment drift', 'unbounded retries', 'silent failures', 'security gaps'],
        'questions': ['What evidence would change the conclusion?', 'Are limitations and alternative explanations explicit?', 'Can another researcher reproduce the main result?'],
    },
}

def get_extended_method_profile(topic: str) -> dict[str, Any]:
    """Return a defensive copy of one extended methodology profile."""
    key = normalize_whitespace(topic).casefold()
    for name, profile in EXTENDED_METHOD_PROFILES.items():
        if name.casefold() == key:
            return {field: list(values) for field, values in profile.items()}
    return {}

def search_extended_method_profiles(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search extended profiles by topic and methods using token overlap."""
    query_tokens = set(tokenize_keywords(query))
    if not query_tokens:
        return []
    ranked = []
    for topic, profile in EXTENDED_METHOD_PROFILES.items():
        searchable = topic + ' ' + ' '.join(profile['methods']) + ' ' + ' '.join(profile['cautions'])
        candidate_tokens = set(tokenize_keywords(searchable))
        score = len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))
        if score > 0:
            ranked.append({'topic': topic, 'score': round(score, 4), **profile})
    ranked.sort(key=lambda row: (-row['score'], row['topic'].casefold()))
    return ranked[:bounded_integer(limit, 20, 1, 100)]

def all_methodology_profile_names() -> list[str]:
    """Return all unique methodology profile names in alphabetical order."""
    return sorted(set(RESEARCH_METHOD_PROFILES) | set(EXTENDED_METHOD_PROFILES), key=str.casefold)

def merged_method_profile(query: str) -> dict[str, Any]:
    """Find a profile across the curated and extended catalogs."""
    base = research_method_profile(query)
    if base.get('matched_topic') and base.get('confidence', 0) >= 0.2:
        return base
    results = search_extended_method_profiles(query, limit=1)
    if results:
        result = dict(results[0])
        result['matched_topic'] = result.pop('topic')
        result['confidence'] = result.pop('score')
        return result
    return base

def methodology_catalog_health() -> dict[str, Any]:
    """Summarize topic catalog coverage and basic schema consistency."""
    combined = list(RESEARCH_METHOD_PROFILES.items()) + list(EXTENDED_METHOD_PROFILES.items())
    required = {'methods', 'evidence', 'cautions', 'questions'}
    invalid = [name for name, profile in combined if set(profile) != required]
    empty = [name for name, profile in combined if any(not profile[k] for k in required)]
    return {'profile_count': len(combined), 'unique_topic_count': len(all_methodology_profile_names()),
            'invalid_profiles': invalid, 'profiles_with_empty_sections': empty,
            'schema_ok': not invalid and not empty}

def self_test_extended_catalog() -> dict[str, Any]:
    """Run deterministic catalog tests without making network requests."""
    health = methodology_catalog_health()
    checks = {
        'catalog_has_topics': health['unique_topic_count'] > 100,
        'catalog_schema': health['schema_ok'],
        'exact_lookup': bool(get_extended_method_profile('AI research tools')),
        'search_returns_results': bool(search_extended_method_profiles('model benchmark', 5)),
        'merged_lookup': bool(merged_method_profile('clinical medicine').get('methods')),
    }
    return {'ok': all(checks.values()), 'checks': checks, 'health': health}



def tokenize_keywords(value: str) -> list[str]:
    """Normalize text into searchable keywords, excluding common stopwords."""
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9+'-]{1,}", str(value or "").casefold())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]




if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink", neutral_hue="slate"),
        css=CSS,
    )
