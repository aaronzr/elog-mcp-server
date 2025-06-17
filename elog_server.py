import httpx
import psycopg
from psycopg.rows import dict_row
from fastmcp import FastMCP

# Can put embedding model on a different machine from chat model used in client
OLLAMA_HOST = "pc100163"      # Aaron's NVIDIA desktop
OLLAMA_EMBED_BASE_URL = f"http://{OLLAMA_HOST}:11434/api/embed"
OLLAMA_EMBED_MODEL = "mxbai-embed-large:latest"

# Postgres connection
POSTGRES_DBNAME = "postgres"
POSTGRES_HOST = "pc100163"    # Aaron's NVIDIA desktop
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres123"
POSTGRES_PORT = 5432

# TODO: Add constants for reranker API connection

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

# Perform vector search on pgvector database
async def vector_search(embedding: list[float]) -> str:
    async with await psycopg.AsyncConnection.connect(
        f"""
            dbname={POSTGRES_DBNAME} 
            user={POSTGRES_USER}
            password={POSTGRES_PASSWORD} 
            host={POSTGRES_HOST}
            port={POSTGRES_PORT}
        """,
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

    # Feel free to change this return type
    return "\n\n".join([f'Event date: {str(row.get("event_date"))} \n \
        {str(row["content"])}\n' for row in rows])


def rerank(query: str, search_results: str) -> str:
    """
    TODO: Pass vector search results and query to a reranker model, e.g.
    `mxbai-rerank-large-v2`. Return a reranked subset of results.

    Args:
        query: A natural language question or phrase.
        search_results: String of vector search results.
    
    Returns:
        Results reranked by relevance to the query.
    """
    # YOUR CODE HERE
    return search_results

@mcp.tool()
async def retrieve_elog(query: str) -> str:
    """
    Search the vector_store for documents semantically similar to the query.
    
    Args:
        query: A natural language question or phrase.
    
    Returns:
        Top 32 results labeled with event dates.
    """
    embedding = await get_embedding(query)
    search_results = await vector_search(embedding)
    return rerank(query, search_results)
    


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