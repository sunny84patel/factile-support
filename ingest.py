"""
ingest_fixed.py - Fixed version with direct Pinecone upsert
- Extracts text + images from a single PDF
- Saves images directly to static/images for web server
- Splits into chunks
- Uses DIRECT Pinecone upsert (bypasses LlamaIndex upload issues)
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import json
import time

# Env loader
from dotenv import load_dotenv

# OpenAI + Pinecone
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# ---------- CONFIG ----------
CHUNK_SIZE = 800     # tokens approx
CHUNK_OVERLAP = 150


# -------------- helpers --------------
def extract_pdf_text_and_images(pdf_path: str, out_image_dir: str):
    """Extract text + images from PDF pages."""
    doc = fitz.open(pdf_path)
    pages = []
    Path(out_image_dir).mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(doc):
        page_num = i + 1
        text = page.get_text("text")
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image.get("ext", "png")
            # Use relative path that matches web server structure
            img_filename = f"page_{page_num}_img_{img_index}.{ext}"
            img_path = os.path.join(out_image_dir, img_filename)
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            # Store just the filename for web server URL construction
            images.append({"img_path": img_filename})
        
        pages.append({"page_num": page_num, "text": text, "images": images})
    
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chunk text into overlapping segments"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    return chunks


def caption_image(img_path: str) -> str:
    """Optional: OCR to get text caption from image."""
    try:
        img = Image.open(img_path)
        text = pytesseract.image_to_string(img).strip()
        return text if text else "Screenshot from documentation"
    except Exception:
        return "Screenshot from documentation"


# -------------- main --------------
def main(args):
    # Load .env file
    load_dotenv()

    pdf_path = args.pdf
    # Save directly to static/images for web server
    out_image_dir = args.out_images or "./static/images"

    pages = extract_pdf_text_and_images(pdf_path, out_image_dir)
    print(f"✅ Extracted {len(pages)} pages")
    
    # Count total images
    total_images = sum(len(p["images"]) for p in pages)
    print(f"📸 Extracted {total_images} images to {out_image_dir}")

    # OpenAI client
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Set OPENAI_API_KEY in .env")
    
    client = OpenAI(api_key=openai_key)
    print("✅ OpenAI client initialized")

    # Pinecone client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    pinecone_index_name = os.getenv("PINECONE_INDEX")
    if not pinecone_index_name:
        raise ValueError("Set PINECONE_INDEX in .env")

    # If index doesn't exist, create it
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if pinecone_index_name not in existing_indexes:
        print(f"📦 Creating Pinecone index: {pinecone_index_name}")
        pc.create_index(
            name=pinecone_index_name,
            dimension=3072,  # text-embedding-3-large
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-2")
        )

    # Connect to Pinecone index
    index = pc.Index(pinecone_index_name)
    namespace = "factile-support"
    
    print(f"✅ Connected to Pinecone index: {pinecone_index_name}")
    print(f"   Namespace: {namespace}")

    # Check stats before
    print("\n📊 Index stats BEFORE upload:")
    stats_before = index.describe_index_stats()
    print(f"   Total vectors: {stats_before.get('total_vector_count', 0)}")

    # Build chunks list
    print("\n📝 Creating document chunks...")
    chunks_to_upload = []
    
    for p in tqdm(pages, desc="Processing pages"):
        page_num = p["page_num"]
        text = p["text"] or ""
        
        if not text.strip():
            continue
            
        text_chunks = chunk_text(text)

        for ci, chunk_content in enumerate(text_chunks):
            chunk_id = f"page_{page_num}_chunk_{ci}"
            metadata = {
                "page": page_num,
                "chunk_index": ci,
                "source": Path(pdf_path).name,
                "text": chunk_content,  # Store text in metadata for retrieval
                "images": [img["img_path"] for img in p["images"]] if p["images"] else []
            }
            chunks_to_upload.append({
                "id": chunk_id,
                "text": chunk_content,
                "metadata": metadata
            })

    print(f"✅ Created {len(chunks_to_upload)} chunks")

    if len(chunks_to_upload) == 0:
        print("❌ No chunks to upload! PDF may be empty or have no text.")
        return

    # Upload to Pinecone in batches
    print(f"\n📤 Uploading {len(chunks_to_upload)} chunks to Pinecone...")
    batch_size = 100
    total_uploaded = 0
    failed_batches = 0

    for i in tqdm(range(0, len(chunks_to_upload), batch_size), desc="Uploading batches"):
        batch = chunks_to_upload[i:i + batch_size]
        texts = [item["text"] for item in batch]
        
        try:
            # Generate embeddings using OpenAI
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            
            # Prepare vectors for Pinecone
            vectors = []
            for j, item in enumerate(batch):
                embedding = response.data[j].embedding
                vectors.append({
                    "id": item["id"],
                    "values": embedding,
                    "metadata": item["metadata"]
                })
            
            # Direct upsert to Pinecone
            index.upsert(vectors=vectors, namespace=namespace)
            total_uploaded += len(vectors)
            
        except Exception as e:
            print(f"\n❌ Batch {i//batch_size + 1} failed: {e}")
            failed_batches += 1
            continue

    print(f"\n✅ Upload complete!")
    print(f"   Uploaded: {total_uploaded} vectors")
    print(f"   Failed batches: {failed_batches}")

    # Wait for Pinecone to process
    print("\n⏳ Waiting 15 seconds for Pinecone to index vectors...")
    time.sleep(15)

    # Check stats after
    print("\n📊 Index stats AFTER upload:")
    stats_after = index.describe_index_stats()
    total_vectors = stats_after.get('total_vector_count', 0)
    print(f"   Total vectors: {total_vectors}")
    
    namespaces = stats_after.get('namespaces', {})
    if namespaces:
        for ns, ns_stats in namespaces.items():
            count = ns_stats.get('vector_count', 0)
            print(f"   Namespace '{ns}': {count} vectors")
    else:
        print("   ⚠️  WARNING: No namespaces found!")

    # Test query to verify upload
    print("\n🔍 Testing query: 'How to create a game'")
    
    try:
        response = client.embeddings.create(
            input=["How to create a game"],
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )
        
        matches = results.get("matches", [])
        if matches:
            print(f"✅ Query successful! Found {len(matches)} matches:")
            for i, match in enumerate(matches[:3], 1):
                score = match.get('score', 0)
                page = match.get('metadata', {}).get('page', 'N/A')
                text_preview = match.get('metadata', {}).get('text', '')[:80]
                print(f"\n   Match {i}:")
                print(f"   Score: {score:.4f} | Page: {page}")
                print(f"   Text: {text_preview}...")
        else:
            print("⚠️  Query returned no matches (data may still be indexing)")
    except Exception as e:
        print(f"❌ Query test failed: {e}")

    # Save manifest for reference
    manifest = {
        "pdf_file": Path(pdf_path).name,
        "total_pages": len(pages),
        "total_images": total_images,
        "total_chunks": len(chunks_to_upload),
        "vectors_uploaded": total_uploaded,
        "vectors_in_index": total_vectors,
        "image_directory": out_image_dir,
        "namespace": namespace,
        "pages": pages[:10]  # Save first 10 pages only
    }
    
    with open("ingest_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ INGESTION COMPLETE!")
    print(f"   📄 {len(pages)} pages processed")
    print(f"   📸 {total_images} images saved to {out_image_dir}")
    print(f"   📦 {total_uploaded} chunks uploaded to Pinecone")
    print(f"   📋 Manifest saved → ingest_manifest.json")
    
    if matches and total_vectors > 0:
        print("\n✅✅✅ SUCCESS! Data is uploaded and queryable! ✅✅✅")
    else:
        print("\n⚠️  Upload completed but verification inconclusive")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF into Pinecone with images")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--out_images", default="./static/images", help="Output directory for images")
    args = parser.parse_args()
    main(args)