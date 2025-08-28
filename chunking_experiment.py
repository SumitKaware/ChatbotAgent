import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import fitz
import pdfplumber
from model import text_splitter, embeddings
from langchain_core.documents import Document as LangchainDocument
from embedding_genration import get_text_embedding
import numpy as np

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

doc_text = "Your long document text here..."  # Replace with actual text or load from PDF
def extract_and_embed_pdf(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Extracts text blocks, tables, and images from a PDF, orders them sequentially,
    generates embeddings, and saves the structured data.

    Args:
        pdf_path (str): The full path to the PDF file.
        output_dir (str): The base directory to save extracted content and embeddings.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an extracted
                              element with its content, description, and embedding,
                              ordered by its appearance in the PDF.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf_output_path = os.path.join(output_dir, pdf_name)
    os.makedirs(pdf_output_path, exist_ok=True)

    print(f"\n--- Processing PDF: {pdf_name} ---")
    chunk_sizes = [300, 500, 700, 1000]
    query = "What funds are allocated to startups?"
    query_embedding = get_text_embedding(query)
    results = []  # List to collect extracted elements as dicts
    page_text_content = None  # Initialize to ensure it's always defined'
    top_sims = []
    try:
        with pdfplumber.open(pdf_path) as pdf_plumber_doc, fitz.open(pdf_path) as fitz_doc:
            num_pages = len(pdf_plumber_doc.pages)
            print(f"  Total pages detected: {num_pages}")

            for page_num in range(num_pages):
                try: # Granular error handling for each page
                    print(f"  Processing Page {page_num + 1}...")
                    plumber_page = pdf_plumber_doc.pages[page_num]
                    fitz_page = fitz_doc[page_num]

                    page_elements = [] # To store elements for current page before sorting

                    # --- Extract Text Blocks and Split ---
                    page_text_content = plumber_page.extract_text(keep_blank_chars=False, layout=True)
                    if page_text_content:
                        # Create a LangchainDocument for the entire page text
                        # Use the full page bbox for all text chunks from this page for sorting
                        page_text_doc = LangchainDocument(
                            page_content=page_text_content,
                            metadata={"page_num": page_num + 1, "source_pdf": pdf_name, "bbox": (0, 0, plumber_page.width, plumber_page.height)}
                        )
                        # Generate embedding for the page text
                        embedding = get_text_embedding(page_text_content)
                        # Append as a dictionary to results
                        results.append({
                            "content": page_text_content,
                            "description": f"Text block from page {page_num + 1}",
                            "embedding": embedding,
                            "metadata": page_text_doc.metadata
                        })
                        for size in chunk_sizes:
                            splitter = text_splitter.__class__(chunk_size=size, chunk_overlap=50)
                            chunks = splitter.split_documents([page_text_doc])
                            similarities = [cosine_sim(get_text_embedding(chunk.page_content), query_embedding) for chunk in chunks]
                            print(f"Chunk size {size}, Top sim: {max(similarities)}")
                            top_sims.append((size, max(similarities)))
                except Exception as e:
                    print(f"    Error processing page {page_num + 1}: {e}")
    except Exception as e:
        print(f"    Error processing page {e}")
    if top_sims:
        # Aggregate by chunk size (if multiple pages, take max per size)
        from collections import defaultdict
        agg = defaultdict(list)
        for size, sim in top_sims:
            agg[size].append(sim)
        avg_sims = {size: max(sims) for size, sims in agg.items()}
        plt.plot(list(avg_sims.keys()), list(avg_sims.values()), marker='o')
        plt.xlabel("Chunk Size")
        plt.ylabel("Top Cosine Similarity")
        plt.title("Chunk Size vs. Top Similarity (Elbow Method)")
        #plt.show()
        plt.savefig(os.path.join(output_dir, "chunk_size_vs_similarity.png"))
    return results
# page_num = 0  # Set to the desired page number (0-based index)
# pdf_name = "example_pdf"  # Set to your PDF file name or desired value
# doc = LangchainDocument(page_content=doc_text, metadata={"page_num": page_num + 1, "source_pdf": pdf_name, "bbox": (0, 0, plumber_page.width, plumber_page.height)})

# doc_dicts = extract_and_embed_pdf("pdfs_to_embed\budget_speech2025.pdf", "output_directory")
# chunk_sizes = [300, 500, 700, 1000]
# query = "What funds are allocated to startups?"
# query_embedding = get_text_embedding(query)

# # Convert each dict to a LangchainDocument
# documents = [
#     LangchainDocument(
#         page_content=d["content"],
#         metadata=d.get("metadata", {})
#     )
#     for d in doc_dicts
# ]

# for size in chunk_sizes:
#     splitter = text_splitter.__class__(chunk_size=size, chunk_overlap=50)
#     chunks = splitter.split_documents(documents)
#     similarities = [cosine_sim(get_text_embedding(chunk.page_content), query_embedding) for chunk in chunks]
#     print(f"Chunk size {size}, Top sim: {max(similarities)}")

results = extract_and_embed_pdf("pdfs_to_embed/budget_speech2025.pdf", "output_directory")
#print(f"Extracted {results} elements from the PDF.")
