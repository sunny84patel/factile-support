"""
query_server.py - IMPROVED with Static UI
- FastAPI app with POST /query
- Serves static HTML UI at root
- Retrieves docs from Pinecone via LlamaIndex
- Calls OpenAI GPT-4o for stepwise answers with image pointers
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import re
from dotenv import load_dotenv
from openai import OpenAI

# LlamaIndex
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Pinecone client
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()

# --- ENV CONFIG ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "factile-support")

if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX]):
    raise EnvironmentError("Missing required environment variables")

# --- Initialize OpenAI client ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# --- Pinecone setup ---
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX)
    logger.info(f"Connected to Pinecone index: {PINECONE_INDEX}")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    raise

# --- Embeddings setup ---
embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.embed_model = embed_model

# --- Vector store ---
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index,
    namespace=PINECONE_NAMESPACE
)

# --- FastAPI app ---
app = FastAPI(title="Factile Support Query API")

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
(static_dir / "images").mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Request/response models ---
class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=6, ge=1, le=20)

class Source(BaseModel):
    page: Optional[int] = None
    id: str
    score: Optional[float] = None

class AnswerOut(BaseModel):
    answer: str
    sources: List[Source] = []
    images: List[str] = []

def humanize_answer(answer: str) -> str:
    """Make GPT response friendlier and less robotic."""
    # Remove markdown asterisks/bullets
    cleaned = re.sub(r"^\s*[\*\-]\s*", "", answer, flags=re.MULTILINE)

    # Replace plain numbered lists with more natural transitions
    lines = cleaned.splitlines()
    friendly_lines = []
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and line[1:3] in [". ", ") "]:
            # Remove raw numbering from GPT
            text = re.sub(r"^\d+[\.\)]\s*", "", line)
            # Add conversational step markers
            if i == 1:
                friendly_lines.append(f"First, {text}")
            elif i == len(lines):
                friendly_lines.append(f"Finally, {text}")
            else:
                friendly_lines.append(f"Next, {text}")
        else:
            friendly_lines.append(line)

    # Join back together
    return "\n".join(friendly_lines).strip()

@app.get("/")
def root():
    """Serve the HTML UI"""
    return FileResponse("static/index.html")


@app.post("/api/query", response_model=AnswerOut)
def query(q: QueryIn):
    """Query the Factile documentation and get GPT-4o generated answers"""
    question = q.question
    top_k = q.top_k
    
    logger.info(f"Received query: {question[:100]}...")

    try:
        # 1) Embed the query
        query_embedding = embed_model.get_text_embedding(question)
        
        # 2) Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE
        )
        
        matches = results.get("matches", [])
        
        if not matches:
            logger.warning("No matching documents found")
            return AnswerOut(
                answer="I couldn't find relevant information in the documentation to answer your question. Please try rephrasing or contact support directly.",
                sources=[],
                images=[]
            )
        
        images_to_return = []
        sources = []
        context_pieces = []

        for match in matches:
            meta = match.get("metadata", {}) or {}
            page = meta.get("page")
            snippet = meta.get("text") or meta.get("content") or ""
            
            if not snippet:
                snippet = f"[See page {page}]"

            context_pieces.append(f"---\nSource: page {page}\n{snippet}\n")

            if "images" in meta and isinstance(meta["images"], list):
                images_to_return.extend([img for img in meta["images"] if img])

            sources.append(Source(
                page=page,
                id=match.get("id", ""),
                score=match.get("score")
            ))

        context = "\n".join(context_pieces)
        
        # Validate context length
        max_context_chars = 12000
        if len(context) > max_context_chars:
            logger.warning(f"Context too long ({len(context)} chars), truncating")
            context = context[:max_context_chars] + "\n...[truncated]"

        # 3) Compose prompt for GPT-4o
        system_prompt = (
            "You are Factile Support Assistant. Use ONLY the provided context from the Factile documentation. "
            "Always answer with clear, numbered step-by-step instructions when appropriate. "
            "If an image is referenced in the context, mention it like: 'See the image on page X for visual guidance.' "
            "Never invent instructions not in the context. If unsure or information is incomplete, "
            "clearly state what you know and what might require additional clarification."
        )

        user_prompt = f"User question: {question}\n\nDocumentation context:\n{context}\n\nProvide a helpful answer based solely on the context above."

        # 4) Call GPT-4o
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.0,
        )

        answer_text = completion.choices[0].message.content.strip()
        answer_text = humanize_answer(answer_text)
        # 5) Process images
        images_unique = list(dict.fromkeys(images_to_return))
        image_urls = []
        
        for img_path in images_unique:
            path_obj = Path(img_path)
            relative_path = f"{path_obj.parent.name}/{path_obj.name}" if path_obj.parent.name else path_obj.name
            image_urls.append(f"/static/images/{relative_path}")
        
        logger.info(f"Query successful: {len(sources)} sources, {len(image_urls)} images")
        
        return AnswerOut(
            answer=answer_text,
            sources=sources,
            images=image_urls
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        pinecone_index.describe_index_stats()
        return {"status": "healthy", "pinecone": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)