import httpx
import psycopg
from psycopg.rows import dict_row
from fastmcp import FastMCP

# Can put embedding model on a different machine from chat model used in client
OLLAMA_EMBED_BASE_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_MODEL = "mxbai-embed-large:latest"
# OLLAMA_EMBED_BASE_URL = "https://vast-2060.tail1f8fd.ts.net"

# Return top k results from vector search
TOP_K = 32

mcp = FastMCP("elog_server")

# Subroutine to embed query using Ollama
async def get_embedding(query: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            OLLAMA_EMBED_BASE_URL,
            json={"model": OLLAMA_EMBED_MODEL, "input": query},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        embedding = response.json()["embeddings"][0]
        assert len(embedding) == 1024
        assert isinstance(embedding[0], float)
        return embedding

@mcp.tool()
async def retrieve_elog(query: str) -> str:
    """
    Search the vector_store for documents semantically similar to the query.
    
    Args:
        query: A natural language question or phrase.
    
    Returns:
        Top 32 results with cosine distances.
    """
    embedding = await get_embedding(query)

    async with await psycopg.AsyncConnection.connect(
        "dbname=postgres user=postgres password=postgres123 host=localhost port=5432",
        row_factory=dict_row
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, content, chunk_id, event_date, mongo_id,
                       embedding <=> %s::vector AS cosine_distance
                FROM vector_store
                ORDER BY cosine_distance ASC
                LIMIT {TOP_K};
                """,
                (embedding,)
            )
            rows = await cur.fetchall()

    return "\n\n".join([f'Event date: {str(row.get("event_date"))} \n \
            {str(row["content"])}\n' for row in rows])


if __name__ == "__main__":
    # Initialize and run the server
    # Initialize FastMCP server on first open port >= 9000
    
    def start_mcp_on_open_port(start_port=9000, max_tries=100):
        for port in range(start_port, start_port + max_tries):
            try:
                mcp.run(
                    port=port,
                    transport='streamable-http',
                    log_level="DEBUG",
                )
            except Exception as e:
                if "Address already in use" in str(e):
                    continue
                else:
                    raise
        raise RuntimeError("No available ports found")
    
    start_mcp_on_open_port()