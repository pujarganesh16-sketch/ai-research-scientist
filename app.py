
import os
import html
import requests
import gradio as gr
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL")
ARXIV_API = "https://export.arxiv.org/api/query"


def get_connection():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is missing. Add it in Render Environment."
        )
    return psycopg.connect(DATABASE_URL, connect_timeout=15)


def initialize_database():
    """Create the application's search history table."""
    if not DATABASE_URL:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_searches (
                    id BIGSERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    result_count INTEGER NOT NULL DEFAULT 0,
                    searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)


def search_papers(query, max_results):
    query = (query or "").strip()

    if not query:
        return "Enter a research topic first.", []

    try:
        response = requests.get(
            ARXIV_API,
            params={
                "search_query": f'all:"{query}"',
                "start": 0,
                "max_results": int(max_results),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=25,
            headers={"User-Agent": "AIResearchScientist/1.0"},
        )
        response.raise_for_status()

        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}

        papers = []
        entries = root.findall("a:entry", ns)

        for entry in entries:
            title = " ".join(
                (entry.findtext("a:title", default="", namespaces=ns)).split()
            )
            summary = " ".join(
                (entry.findtext("a:summary", default="", namespaces=ns)).split()
            )
            link = entry.findtext("a:id", default="", namespaces=ns)
            published = entry.findtext(
                "a:published", default="", namespaces=ns
            )[:10]

            authors = [
                a.findtext("a:name", default="", namespaces=ns)
                for a in entry.findall("a:author", ns)
            ]

            papers.append({
                "Title": title,
                "Authors": ", ".join(authors[:8]),
                "Published": published,
                "Abstract": summary,
                "Link": link,
            })

        if DATABASE_URL:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO research_searches (query, result_count)
                        VALUES (%s, %s)
                        """,
                        (query, len(papers)),
                    )

        if not papers:
            return f"No papers found for: {query}", []

        message = f"Found {len(papers)} papers for: {query}"
        return message, papers

    except requests.RequestException as e:
        return f"Paper search failed. Please try again. Details: {e}", []
    except Exception as e:
        return f"An error occurred: {e}", []


def show_history():
    if not DATABASE_URL:
        return "Database is not configured."

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT query, result_count, searched_at
                    FROM research_searches
                    ORDER BY searched_at DESC
                    LIMIT 20
                """)
                rows = cur.fetchall()

        if not rows:
            return "No searches have been saved yet."

        lines = ["Recent searches:"]
        for query, count, when in rows:
            lines.append(
                f"- {when:%Y-%m-%d %H:%M UTC} | {query} | {count} papers"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Could not load history: {e}"


def check_database():
    if not DATABASE_URL:
        return "DATABASE_URL is not configured in Render."

    try:
        initialize_database()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT current_database()")
                database = cur.fetchone()[0]
        return f"Connected successfully to Neon database: {database}"
    except Exception as e:
        return f"Database check failed: {e}"


with gr.Blocks(title="AI Research Scientist") as demo:
    gr.Markdown(
        """
        # AI Research Scientist
        Search scientific papers, read abstracts, and explore research topics.

        Paper metadata and abstracts are retrieved from arXiv.
        """
    )

    with gr.Row():
        topic = gr.Textbox(
            label="Research topic",
            placeholder="e.g. artificial intelligence in healthcare",
            scale=4,
        )
        limit = gr.Dropdown(
            choices=[5, 10, 20, 30],
            value=10,
            label="Maximum papers",
            scale=1,
        )

    search_button = gr.Button("Search Research Papers", variant="primary")
    status = gr.Markdown()
    results = gr.JSON(label="Research Papers")

    search_button.click(
        fn=search_papers,
        inputs=[topic, limit],
        outputs=[status, results],
    )
    topic.submit(
        fn=search_papers,
        inputs=[topic, limit],
        outputs=[status, results],
    )

    gr.Markdown("## Database")
    db_button = gr.Button("Check Neon Connection")
    db_status = gr.Textbox(label="Database status", lines=2)
    db_button.click(fn=check_database, outputs=db_status)

    gr.Markdown("## Recent Search History")
    history_button = gr.Button("Show Recent Searches")
    history = gr.Textbox(label="Saved searches", lines=10)
    history_button.click(fn=show_history, outputs=history)

    gr.Markdown(
        "Research source: [arXiv](https://arxiv.org/)"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)







