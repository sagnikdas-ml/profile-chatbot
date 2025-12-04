import io
import math
import textwrap

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader
from google import genai

# ==============================
# CONFIG
# ==============================
CHAT_MODEL = "gemini-2.5-flash"     # Good price/performance for chat
EMBED_MODEL = "text-embedding-004"  # If this errors, try "gemini-embedding-001"
DEFAULT_CV_URL = (
    "https://github.com/sagnik-sudo/cv-sagnikdas/blob/main/Sagnik%20Das%20Lebenslauf.pdf"
)

st.set_page_config(
    page_title="Sagnik CV Chatbot",
    page_icon="📄",
    layout="wide",
)


# ==============================
# HELPERS
# ==============================

def github_to_raw(url: str) -> str:
    """Convert a GitHub 'blob' URL to a raw.githubusercontent.com URL."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def create_client(api_key: str) -> genai.Client:
    """Create Gemini client for Developer API (AI Studio)."""
    return genai.Client(api_key=api_key)


def fetch_pdf_bytes(url: str) -> bytes:
    """Download PDF bytes from a (raw) URL."""
    raw_url = github_to_raw(url)
    resp = requests.get(raw_url)
    resp.raise_for_status()
    return resp.content


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF (simple, linear)."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    full_text = "\n\n".join(pages_text)
    return full_text


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100):
    """Naive text chunking by characters with overlap."""
    text = text.strip().replace("\r", " ")
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # step with overlap
        if start < 0:
            start = 0
    return chunks


def embed_chunks(client: genai.Client, chunks):
    """Embed a list of text chunks using Gemini embeddings."""
    # Gemini embeddings API can take a list of contents in one call
    # https://ai.google.dev/gemini-api/docs/embeddings  [oai_citation:0‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/embeddings?utm_source=chatgpt.com)
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
    )
    # Result can be either a single or batch embeddings structure; handle batch case.
    # google-genai returns resp.embeddings as a list in batch mode.  [oai_citation:1‡PyPI](https://pypi.org/project/google-genai/?utm_source=chatgpt.com)
    vectors = [np.array(e.values, dtype=np.float32) for e in resp.embeddings]
    return vectors


def embed_query(client: genai.Client, query: str):
    resp = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
    )
    emb = np.array(resp.embeddings[0].values, dtype=np.float32)
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def retrieve_relevant_chunks(query_emb, chunk_embs, chunks, top_k: int = 4):
    """Return top-k (chunk, score) pairs."""
    scores = []
    for emb, txt in zip(chunk_embs, chunks):
        score = cosine_similarity(query_emb, emb)
        scores.append((score, txt))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


def build_rag_prompt(user_message: str, retrieved_chunks):
    """Compose a prompt with context from CV and the user question."""
    context_blocks = []
    for i, (score, chunk) in enumerate(retrieved_chunks, start=1):
        header = f"[CV snippet {i} | similarity={score:.3f}]"
        ctx = textwrap.indent(chunk.strip(), prefix="  ")
        context_blocks.append(f"{header}\n{ctx}")

    context_text = "\n\n".join(context_blocks) if context_blocks else "No CV text available."

    system_instruction = (
        "You are a helpful assistant that answers questions ONLY using the provided CV of "
        "Sagnik Das. If the answer is not clearly in the CV context, say you don't know "
        "or that it isn't mentioned in the CV. Be concise and professional."
    )

    full_prompt = f"""{system_instruction}

[CV CONTEXT START]
{context_text}
[CV CONTEXT END]

Now answer the user's question strictly based on this CV.

User question: {user_message}
"""
    return full_prompt


def ask_gemini(client: genai.Client, history, prompt: str) -> str:
    """
    Very simple: we just send the prompt as a single text input.
    `history` isn't used here (you could prepend it if you want multi-turn memory).
    """
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# ==============================
# SESSION STATE INIT
# ==============================
if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "client" not in st.session_state:
    st.session_state.client = None

if "cv_loaded" not in st.session_state:
    st.session_state.cv_loaded = False

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "cv_chunks" not in st.session_state:
    st.session_state.cv_chunks = []

if "cv_embeddings" not in st.session_state:
    st.session_state.cv_embeddings = []

if "messages" not in st.session_state:
    st.session_state.messages = []  # chat messages


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("📄 About this app")
    st.markdown(
        """
This app lets you **chat with Sagnik's CV**.

- Uses **Google Gemini** (via AI Studio)  
- Does **RAG** over a CV PDF hosted on GitHub  
- Answers questions **strictly from the CV**  
        """
    )
    st.markdown("---")
    st.markdown("**Tips**")
    st.markdown(
        "- Ask things like:\n"
        "  - `What technologies does Sagnik know?`\n"
        "  - `Summarize Sagnik's experience in 3 bullet points.`\n"
        "  - `What languages does Sagnik speak?`"
    )


# ==============================
# MAIN LAYOUT
# ==============================
st.title("📄 Sagnik CV Chatbot")
st.caption("Backed by Google Gemini + RAG over your GitHub CV")

# Top status cards
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API Key", "Set" if st.session_state.api_key else "Missing")
with col2:
    st.metric("CV Loaded", "Yes" if st.session_state.cv_loaded else "No")
with col3:
    st.metric("Chunks", len(st.session_state.cv_chunks))


# ==============================
# STEP 1: API KEY + CV URL FORM
# ==============================
if not st.session_state.api_key:
    st.subheader("Step 1 · Enter your Gemini API key")

    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "Google AI Studio API key",
            type="password",
            help="Create a key in Google AI Studio and paste it here.",
        )
        cv_url_input = st.text_input(
            "CV PDF URL (GitHub or raw)",
            value=DEFAULT_CV_URL,
            help="You can change this if you move your CV.",
        )
        submitted = st.form_submit_button("Load CV & Continue")

    if submitted:
        if not api_key_input:
            st.error("API key is required.")
        else:
            try:
                client = create_client(api_key_input)
                pdf_bytes = fetch_pdf_bytes(cv_url_input)
                cv_text = extract_text_from_pdf(pdf_bytes)
                chunks = chunk_text(cv_text)

                if not chunks:
                    st.error("No text extracted from the CV PDF. Check the file.")
                else:
                    # Embed chunks
                    with st.spinner("Embedding CV chunks with Gemini..."):
                        chunk_embs = embed_chunks(client, chunks)

                    # Save in session
                    st.session_state.api_key = api_key_input
                    st.session_state.client = client
                    st.session_state.cv_loaded = True
                    st.session_state.cv_text = cv_text
                    st.session_state.cv_chunks = chunks
                    st.session_state.cv_embeddings = chunk_embs

                    st.success(f"CV loaded and indexed ({len(chunks)} chunks).")
                    st.rerun()

            except Exception as e:
                st.error(f"Failed to load CV or create client: {e}")

    st.stop()  # Don't show chat until key is set & CV is (at least attempted)


# If we're here, API key exists; ensure client + CV are ready
if st.session_state.client is None:
    try:
        st.session_state.client = create_client(st.session_state.api_key)
    except Exception as e:
        st.error(f"Error recreating Gemini client: {e}")
        st.stop()

client = st.session_state.client

# Small panel to reload CV if needed
with st.expander("🔄 Reload / change CV source"):
    new_cv_url = st.text_input(
        "CV PDF URL",
        value=DEFAULT_CV_URL,
        key="cv_url_reload",
        help="Change and click the button below to re-index.",
    )
    if st.button("Re-download & re-index CV"):
        try:
            pdf_bytes = fetch_pdf_bytes(new_cv_url)
            cv_text = extract_text_from_pdf(pdf_bytes)
            chunks = chunk_text(cv_text)

            if not chunks:
                st.error("No text extracted from the CV PDF. Check the file.")
            else:
                with st.spinner("Embedding CV chunks with Gemini..."):
                    chunk_embs = embed_chunks(client, chunks)

                st.session_state.cv_loaded = True
                st.session_state.cv_text = cv_text
                st.session_state.cv_chunks = chunks
                st.session_state.cv_embeddings = chunk_embs

                st.success(f"Re-indexed CV ({len(chunks)} chunks).")
                st.rerun()
        except Exception as e:
            st.error(f"Error reloading CV: {e}")

# Optional: show extracted text preview
with st.expander("👀 Preview extracted CV text"):
    preview = st.session_state.cv_text[:2000]
    st.text(preview if preview else "No text extracted.")


st.markdown("---")
st.subheader("Step 2 · Chat with the CV")


# ==============================
# CHAT UI
# ==============================
# Show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask something about Sagnik's background, skills, etc.")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not st.session_state.cv_loaded or not st.session_state.cv_chunks:
        with st.chat_message("assistant"):
            st.error("CV is not loaded yet. Please reload the CV from the expander above.")
    else:
        try:
            with st.chat_message("assistant"):
                with st.spinner("Searching CV and thinking..."):
                    # 1) Embed query
                    q_emb = embed_query(client, user_input)

                    # 2) Retrieve top-k chunks
                    retrieved = retrieve_relevant_chunks(
                        q_emb,
                        st.session_state.cv_embeddings,
                        st.session_state.cv_chunks,
                        top_k=4,
                    )

                    # 3) Build RAG prompt
                    rag_prompt = build_rag_prompt(user_input, retrieved)

                    # 4) Ask Gemini
                    answer = ask_gemini(client, st.session_state.messages, rag_prompt)
                    st.markdown(answer)

            # Save assistant response
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"Error during RAG or Gemini call: {e}")