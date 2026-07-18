# Loan Advisory RAG Agent

An AI-powered loan advisory agent that lets users ask loan-related questions in natural language and get accurate, source-backed answers — grounded in real bank policy documents and RBI regulatory guidelines, not general LLM knowledge.

It retrieves the most relevant passages from a corpus of loan policy PDFs, reranks them for relevance, and generates a response that cites its sources — covering eligibility, EMIs, documentation, foreclosure charges, and RBI rules. Users can also upload their own PDF/TXT/DOCX documents on the fly and immediately ask questions about them.

## Why this exists

Loan policy documents are long, inconsistent across banks, and easy to misread. A chatbot that answers loan questions from general knowledge alone will confidently hallucinate numbers. This project is built around the opposite principle: **the system should refuse to answer rather than guess**, and every claim it makes should be traceable back to a specific document and section.

## Key features

- **Hybrid retrieval** — dense embedding search (FAISS + `bge-small-en-v1.5`) fused with BM25 keyword search via Reciprocal Rank Fusion, so both semantic queries and exact-term queries (e.g. specific CIBIL scores) retrieve well.
- **Cross-encoder reranking** — a second-stage `ms-marco-MiniLM` reranker reorders candidates by actual query relevance before they reach the LLM.
- **Grounded generation with citations** — every answer is generated strictly from retrieved source excerpts, with inline `[1]`, `[2]` citations mapped to bank, document, and section.
- **Confidence-gated refusal** — if retrieval confidence falls below a threshold, or a query names a bank/product not in the corpus, the system explicitly says so instead of guessing.
- **Conflict and uncertainty handling** — when sources disagree or a document flags a value as unconfirmed, the system surfaces the disagreement instead of averaging or picking one answer.
- **Bank-isolated comparisons** — "compare X vs Y" queries retrieve separately per bank so one bank's chunks can't crowd out another's.
- **Live document uploads** — users can attach a PDF/TXT/DOCX in the chat UI; it's chunked, embedded, and merged into the live index so it becomes queryable in the same session, alongside the base corpus.
- **Built-in EMI calculator** — rule-based shortcut that detects EMI calculation requests and computes them deterministically, without going through the LLM.
- **Fully local inference** — runs on a local open-weight LLM (`Qwen2.5-1.5B-Instruct`), so no document content or query leaves the machine via an external API.

## Architecture

![Architecture](screenshots/architecture.png)

```
User Query → Filter Extraction (bank/product) → Dense Retrieval (FAISS) + BM25 Retrieval
           → Reciprocal Rank Fusion → Cross-Encoder Reranking
           → Confidence Check ── below threshold → Refuse
           → Grounded Generation (local LLM, strict system prompt) → Answer + Citations
```

User-uploaded documents feed into the same pipeline: extracted → chunked → embedded → added to the live FAISS/BM25 indexes → immediately retrievable through the same flow above.

## Tech stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) |
| Vector index | `faiss-cpu` |
| Keyword search | `rank-bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `Qwen2.5-1.5B-Instruct` (local, via `transformers`) |
| PDF parsing | `pdfplumber` |
| UI | `gradio` (multimodal chat interface) |

## Project structure
```
Loan-Advisory-RAG-Agent/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── Loan Advisory RAG Pipeline.ipynb        # end-to-end pipeline — run top to bottom
│
├── real_indian_banks_full_corpus.zip       # HDFC, ICICI, PNB, SBI, Axis policy PDFs
├── rbi_regulatory_reference_docs.zip       # RBI regulatory reference docs
│
├── chatbot.png                             # Gradio chat UI in action
├── Retrieval.png                           # hybrid retrieval output
├── Retrival answeing a query.png           # end-to-end query → answer example
│
└── vector_store/
      └── .gitkeep                          # keeps the folder tracked; FAISS/BM25
                                             # artifacts generated here are gitignored
```
 

> **Screenshot note:** `chatbot.png` and `retrieval.png` are placeholders generated for this scaffold — they were not captured from a live run. Run the notebook's Gradio cell, take real screenshots of the chat UI and a retrieval/citation result, and replace these two files before presenting the project.

## Setup

```bash
git clone https://github.com/<your-username>/Loan-Advisory-RAG-Agent.git
cd Loan-Advisory-RAG-Agent
pip install -r requirements.txt
```

Then open `Loan_Advisory_RAG.ipynb` in Jupyter or Google Colab and run top to bottom.

- **Step 1b** prompts you to upload two zip files — use `sample_documents/real_indian_banks_full_corpus.zip` and `sample_documents/rbi_regulatory_reference_docs.zip`.
- Steps 2-4 build the chunked corpus, embeddings, FAISS index, and BM25 index (optionally save these into `vector_store/` for reuse).
- Steps 6-8 load the reranker and local LLM and wire up `answer_query()`.
- Step 9b/9c add live document-upload ingestion.
- Step 10 launches the Gradio chat UI, with a paperclip icon for attaching your own PDF/TXT/DOCX.

A GPU runtime is recommended for the LLM generation step but not required.

## Usage

Example queries:
- "What is the minimum CIBIL score for an HDFC home loan?"
- "Compare home loan interest rates of SBI and HDFC."
- "Calculate EMI for 10 lakh at 8.5% for 20 years"
- "What is RBI's penal charge policy?"

Or attach your own document via the chat UI's paperclip icon and ask a question about it directly.

## Data

`sample_documents/` contains the full corpus used to build and validate this project: policy documents for five Indian banks (HDFC, ICICI, PNB, SBI, Axis) covering home, personal, car, education, and gold loans, plus RBI regulatory reference documents (fair practices code, penal charges, EBLR framework, gold lending, digital lending guidelines, and the ombudsman scheme). These are bundled as the two zip files Step 1b expects, with a handful of individual PDFs also included for quick browsing.

## Limitations

- Answer quality is bounded by the small local LLM (1.5B parameters); a larger model would improve reasoning on multi-condition queries.
- Comparison-mode retrieval currently filters strictly by named bank/RBI, so user-uploaded documents aren't included in "compare X vs Y" queries — only in single-source queries.
- No formal retrieval/answer evaluation harness yet (planned — see below).
- BM25 index is rebuilt on every upload rather than updated incrementally; fine at current corpus scale, but would need optimization at larger scale.

## Roadmap

- [ ] Evaluation harness with labeled Q&A pairs (retrieval precision/recall + answer correctness)
- [ ] Extend comparison-mode retrieval to include user uploads
- [ ] Swap in a larger local LLM or add an optional hosted-API fallback
- [ ] Persist per-session uploaded documents with explicit user-controlled deletion
- [ ] Replace placeholder screenshots with real captures from a live run

## License

MIT — see [LICENSE](LICENSE).
