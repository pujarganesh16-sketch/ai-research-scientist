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
    summarize_button.click(ai_summary, inputs=[papers_state, query], outputs=ai_output)
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

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3, max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink", neutral_hue="slate"),
        css=CSS,
    )
