import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_stride_kernel"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_set_to_1d"] = "0"

from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from google.colab import drive

import torch
import csv
import tempfile
import fitz
import pathlib
import os
import faiss

def generate_answer(query, chunks, tokenizer, llm):
    context = "\n\n".join([f"[Page {c['page_id']}]\n{c['text']}" for c in chunks[:5]])
 
    prompt = f"""
    You are a document question answering system.

    Answer the question ONLY using the provided context.

    Rules:
    - Do NOT use outside knowledge.
    - Do NOT guess.
    - Do NOT infer missing values.
    - If the answer cannot be found exactly, output:
    Not found in document
    - Keep the answer concise.
    - Do NOT explain reasoning.
    - Do NOT repeat the question.
    - For list questions, output only the list items.
    - For numeric questions, output only the final value and unit if present.
    - For comparison questions, output only the final result.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(llm.device)

    with torch.no_grad():
            output = llm.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.0,
                do_sample=False
            )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("Answer:")[-1].strip()

def convert_to_chunks( result: list[dict], file_id: str, chunk_size: int = 400, overlap: int = 120 ) -> list[dict]:

    chunks = []
    for page in result:
        text = page.get("text", "")
        page_id = page.get("page_id")

        if not text or not isinstance(text, str):
            continue

        words = text.split()

        start = 0
        chunk_id = 0

        while start < len(words):

            end = start + chunk_size

            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "page_id": page_id,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "file_id": file_id
            })

            chunk_id += 1
            start += chunk_size - overlap  #sliding window

    return chunks

def check_file_type(document_ : list[dict] ) -> bool:
    if not isinstance(document_, list) or len(document_) == 0:
        return False

    for page in document_:
        if not isinstance(page, dict):
            return False
        if "page_id" not in page or "text" not in page:
            return False
        if not isinstance(page["text"], str):
            return False
        if not isinstance(page["page_id"], int):
            return False

    return True
        
def get_pages(path_: str, file_id: str, ocr) -> list[dict]:
    try:
        results_ = []
        with fitz.open(str(path_)) as doc:
            for page_number, page in enumerate(doc, start=1):
                pix = page.get_pixmap()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img_path = tmp.name
                pix.save(img_path)

                try:
                    ocr_result = ocr.ocr(img_path)
                    if not ocr_result or not ocr_result[0]:
                        continue

                    page_text = " ".join(
                        line[1][0] for line in ocr_result[0] if line and line[1]
                    )

                    results_.append({
                        "page_id": page_number,
                        "text": page_text,
                        "file_id": file_id
                    })
                finally:
                    if os.path.exists(img_path):
                        os.remove(img_path)

        return convert_to_chunks(results_, file_id)

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")
        
def embed_chunks_and_indexing(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, list[dict]]:
    #chunks is the original list of dicts with page_id, chunk_id and text.
    data_ = [chunk["text"] for chunk in chunks]

    #Generating embeddings for chunks
    chunk_embeds = model.encode(
    data_,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
    )
    
    #Forming Faiss index
    dimen_ = chunk_embeds.shape[1]
    index = faiss.IndexFlatIP(dimen_)
    index.add(chunk_embeds)

    tokenized_chunks = [chunk["text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    #Returning the index
    return index, chunks, bm25
    
def extract_file_id_and_query(path_ : pathlib.Path) -> list[dict]:
    requirements = []

    with open(path_, mode="r", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:
            requirements.append({
                "file_id": row.get("file_id"),
                "query": row.get("question")
            })
    
    return requirements

def embed_query_and_search(query: str , file_id,  DB: dict, top_k : int = 5):
    relevant_db = DB[file_id]

    query_embed = model.encode(
        [query],
        show_progress_bar = True,
        convert_to_numpy = True,
        normalize_embeddings = True
    )

    scores, indices = relevant_db["index"].search(query_embed, top_k)

    results = [
        relevant_db["chunks"][i]
        for i in indices[0]
    ]

    pages = sorted(set(r["page_id"] for r in results))
    answer = results[0]["text"] if results else ""

    return {
        "answer": answer,
        "pages": pages,
        "results": results,
        "file_id": file_id
        }

def keyword_search(query: str, db, top_k: int = 10):
    tokenized_query = query.split()

    scores = db["bm25"].get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True)[:top_k]

    return [db["chunks"][i] for i in ranked_indices]

def rerank(query, chunks):
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, chunks), reverse=True)
    return [c for _, c in ranked[:5]]

if __name__ == "__main__":
    drive.mount('/content/drive')
    path_consider_reader = pathlib.Path("/content/drive/MyDrive/file_path_for_reader")
    path_consider_query = pathlib.Path("/content/drive/MyDrive/file_path_for_query")
    
    Vector_DB = {}

    required = extract_file_id_and_query(path_consider_query)

    # collect unique file_ids from CSV
    necessary_files = set()
    available_pdfs = {f.stem: f for f in path_consider_reader.glob("*.pdf")}

    for item in required:
        fid = item["file_id"]

        if fid and fid in available_pdfs:
            necessary_files.add(fid)

    model = SentenceTransformer(
    "BAAI/bge-m3",
    device="cpu"
    )

    reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    device="cpu"
    )

    llm_model_name = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    llm = AutoModelForCausalLM.from_pretrained(
      llm_model_name,
      torch_dtype=torch.float16,
      device_map="cuda"
    )

    for file_id in necessary_files:
        file_path = available_pdfs[file_id]

        if file_id.startswith("j_"):
            ocr = PaddleOCR(lang="japan")

        elif file_id.startswith("v_"):
            ocr = PaddleOCR(lang="vi")

        else:
            continue

        pages = get_pages(file_path, file_id, ocr)

        if check_file_type(pages):
            Faiss_embed, og_txt, bm25 = embed_chunks_and_indexing(pages)

            Vector_DB[file_id] = {
                "index": Faiss_embed,
                "chunks": og_txt,
                "bm25": bm25,
                "language": "ja" if file_id.startswith("j_") else "vi"
            }

            print(f"Built DB for {file_id}")
        else:
            print(f"Something went wrong with {file_id}")

    # run all queries against the built DB
    for item in required:
        query = item.get("query")
        fid = item.get("file_id")

        if not query or not fid or fid not in Vector_DB:
            continue

        embed_result = embed_query_and_search(query, fid, Vector_DB, top_k=20)
        semantic_result = keyword_search(query, Vector_DB[fid], top_k=20)

        combined = embed_result["results"] + semantic_result

        seen = set()
        unique_chunks = []

        for chunk in combined:
            key = (chunk["page_id"], chunk["chunk_id"])

            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        reranked = rerank(query, unique_chunks)

        answer = generate_answer(query, reranked, tokenizer, llm)
        pages = sorted(set(c["page_id"] for c in reranked))

        print({
          "answer": answer,
          "pages": pages
        })