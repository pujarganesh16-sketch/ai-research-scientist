
import os
import gradio as gr
import psycopg

DATABASE_URL = os.environ.get("DATABASE_URL")

def check_database():
    if not DATABASE_URL:
        return "DATABASE_URL is not configured in Render."
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=15) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                tables = [row[0] for row in cur.fetchall()]
        return "Connected to Neon successfully.\n\nTables found: " + str(len(tables)) + "\n\n" + "\n".join(tables)
    except Exception as e:
        return "Database connection failed: " + str(e)

with gr.Blocks(title="AI Research Scientist") as demo:
    gr.Markdown("# AI Research Scientist")
    gr.Markdown("Research workspace connected to your persistent Neon database.")
    check = gr.Button("Check database connection")
    result = gr.Textbox(label="Database status", lines=12)
    check.click(fn=check_database, outputs=result)

port = int(os.environ.get("PORT", "7860"))
demo.launch(server_name="0.0.0.0", server_port=port)






