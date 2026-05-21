import os

os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_stride_kernel"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_set_to_1d"] = "0"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi
from google.colab import drive
from pyvi import ViTokenizer

import torch
import csv
import tempfile
import fitz
import pathlib
import faiss
import fugashi

def keyword_search(query: str, db, top_k: int = 10):

    tokenized_query = tokenize_text(
        query,
        db["language"]
    )

    scores = db["bm25"].get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [
        db["chunks"][i]
        for i in ranked_indices
    ]

def embed_chunks_and_indexing(
    chunks: list[dict],
    language: str
):

    # EMBEDDING INPUT
    data_ = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    # EMBEDDINGS
    chunk_embeds = model.encode(
        data_,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS INDEX
    dimen_ = chunk_embeds.shape[1]

    index = faiss.IndexFlatIP(dimen_)

    index.add(chunk_embeds)

    # BM25 TOKENIZATION
    tokenized_chunks = [tokenize_text(chunk["text"], language)
        for chunk in chunks]


    bm25 = BM25Okapi(tokenized_chunks)

    return index, chunks, bm25

def extract_file_id_and_query(path_: pathlib.Path) -> dict:

    with open(path_, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "file_id": row.get("file_id"),
                "query": row.get("question")
            }

def tokenize_text(text: str, language: str):

    if language == "ja":
        tokens = [
            word.surface
            for word in jp_tagger(text)
        ]
        return tokens

    elif language == "vi":
        tokenized = ViTokenizer.tokenize(text)
        return tokenized.split()

    return text.lower().split()

def embed_query_and_search(
    query: str,
    file_id,
    DB: dict,
    top_k: int = 10
):

    relevant_db = DB[file_id]

    query_embed = model.encode(
        [f"query: {query}"],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = relevant_db["index"].search(
        query_embed,
        top_k
    )

    results = [
        relevant_db["chunks"][i]
        for i in indices[0]
    ]

    return {
        "results": results,
        "file_id": file_id
    }

def rerank(query, chunks):

    if not chunks:
        return []

    pairs = [
        (query, chunk["text"])
        for chunk in chunks
    ]

    scores = reranker.predict(
        pairs,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(scores, chunks),
        key=lambda x: x[0],
        reverse=True
    )

    return [c for _, c in ranked[:10]]

def generate_answer(query, chunks, tokenizer, llm):

    context = "\n\n".join(
        [f"[Page {c['page_id']}]\n{c['text']}" for c in chunks[:5]]
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict document QA system.\n"
                "Answer ONLY using the provided context.\n\n"

                "Rules:\n"
                "- Extract ONLY the exact answer span from context\n"
                "- Never summarize\n"
                "- Never continue sentences\n"
                "- No explanations\n"
                "- No reasoning\n"
                "- No chain of thought\n"
                "- No extra text\n"
                "- Keep answers extremely short\n"
                "- Output only the final answer\n"
                "- If answer missing output exactly:\n"
                "Not found in document"
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question:\n{query}"
            )
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(llm.device)

    with torch.no_grad():

        output = llm.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=0.0
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return answer

if __name__ == "__main__":
    drive.mount('/content/drive') #<--- mounting drive.

    path_consider_reader = pathlib.Path("/content/drive/MyDrive/path_to_all_train_files") #<--- training files are located here.

    path_consider_query = pathlib.Path("/content/drive/MyDrive/path_to_all_queries_in_csv") #<--- query csv is here.

    output_csv = pathlib.Path( "/content/drive/MyDrive/submission.csv") #<--- output csv here.

    # =====================================================
    # LOAD MODELS
    # =====================================================

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base", #<--- embedding model
        device="cpu"
    )

    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3", #<--- reranking model
        device="cpu"
    )

    llm_model_name = "Qwen/Qwen2.5-3B-Instruct" #<--- llm reasoning model

    tokenizer = AutoTokenizer.from_pretrained(llm_model_name) #<--- tokeniation model

    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    jp_tagger = fugashi.Tagger()


    # ocr
    #ocr_jp = PaddleOCR(lang="japan")
    #ocr_vi = PaddleOCR(lang="vi")

    # cache
    Vector_DB = {}
    #chunks = None

    # =====================================================
    # output csv file
    # =====================================================

    with open(  #<--- csv file opening
        output_csv,
        mode="w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter( #<--- csv file writer
            csv_file,
            fieldnames=["file_id", "answer", "pages"]
        )

        writer.writeheader() #<--- writing the header
        language = None
        # pdf map
        available_pdfs = {
            f.stem: f
            for f in path_consider_reader.glob("*.pdf")} #<--- consider all available pdf files

        for output in extract_file_id_and_query(path_consider_query): #<--- we extract which file and query we want from the original file, output: {file_id: j_001, query: some query}.
            fid = output["file_id"] #<--- file_id for the current row
            query = output["query"] #<--- query for the current row

            if not query or not fid: #<--- check for not empty.
                continue

            if fid not in available_pdfs: #check for file present in directory.
                continue

            if fid not in Vector_DB:
                file_path = available_pdfs[fid] #temp_holding file path
                loader = PyPDFLoader(str(file_path)) #<-- opening the said file
                doc = loader.load() #<--- loading said file

                if fid.startswith("j_"):

                    text_splitter = RecursiveCharacterTextSplitter( #<--- These are basic chunks defined by the inputs we provide
                    chunk_size=500,
                    chunk_overlap=80,
                    length_function=len,
                    keep_separator=True,
                    separators = [
                      "\n\n",   # paragraph
                      "\n",     # line break
                      "。",     # sentence end
                      "! ",
                      "? ",
                      "、",     # comma-like pause
                      ""])

                    language = "ja"

                elif fid.startswith("v_"):

                    text_splitter = RecursiveCharacterTextSplitter( #<--- These are basic chunks defined by the inputs we provide
                    chunk_size=500,
                    chunk_overlap=80,
                    length_function=len,
                    keep_separator=True,
                    separators=[
                      "\n\n",
                      "\n",
                      ". ",
                      "! ",
                      "? ",
                      ", ",
                      " ",
                      ""])

                    language = "vi"

                else:
                    continue

                chunks = []

                splits = text_splitter.split_documents(doc) #<--- actual chunk formation

                for idx, chunk in enumerate(splits): #<--- data extraction
                    text = chunk.page_content.strip()

                    if not text:
                        continue

                    chunks.append({
                        "page_id": chunk.metadata.get("page", 0) + 1,
                        "chunk_id": idx,
                        "text": chunk.page_content,
                        "file_id": fid})

                index, chunks, bm25 = embed_chunks_and_indexing(chunks, language)

                Vector_DB[fid] = {
                    "index": index,
                    "chunks": chunks,
                    "bm25": bm25,
                    "language": language
                    }

                print(f"Built DB for {fid}")

            semantic_results = embed_query_and_search(
                query,
                fid,
                Vector_DB,
                top_k=25
                )

            keyword_results = keyword_search(
                query,
                Vector_DB[fid],
                top_k=25 )

            combined = (
                semantic_results["results"]
                + keyword_results)

            seen = set()

            unique_chunks = []

            for chunk in combined:

                key = (
                        chunk["page_id"],
                        chunk["chunk_id"])

                if key not in seen:

                    seen.add(key)

                    unique_chunks.append(chunk)

                # RERANK
            reranked = rerank(
                    query,
                    unique_chunks
            )

                # GENERATE ANSWER
            answer = generate_answer(
                    query,
                    reranked,
                    tokenizer,
                    llm
            )

                # PAGE IDS
            if reranked:
              pages = [reranked[0]["page_id"]]

            print({
                    "file_id": fid,
                    "answer": answer,
                    "pages": pages
                })

                # WRITE CSV
            writer.writerow({
                    "file_id": fid,
                    "answer": answer,
                    "pages": " ".join(map(str, pages))
            })

    print("DONE")