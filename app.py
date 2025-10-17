import os
import json
from typing import List, Dict, Optional, Tuple
import hashlib
import pathlib
import re
import glob
from dataclasses import dataclass

import streamlit as st
from openai import OpenAI

# ---------------- Provider config ----------------
PROVIDER = os.getenv("PROVIDER", "GROQ").upper()  # GROQ | TOGETHER | OPENROUTER | FIREWORKS | DEEPINFRA

# Map provider -> base_url + a sensible default model name
PROVIDERS = {
    "GROQ": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        "key_env": "GROQ_API_KEY",
        "secret_key": "GROQ_API_KEY",
        "extra_headers": {},
    },
    "TOGETHER": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "key_env": "TOGETHER_API_KEY",
        "secret_key": "TOGETHER_API_KEY",
        "extra_headers": {},
    },
    "OPENROUTER": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"),
        "key_env": "OPENROUTER_API_KEY",
        "secret_key": "OPENROUTER_API_KEY",
        "extra_headers": {
            # "HTTP-Referer": "https://your-app-url",
            # "X-Title": "ESCP ChatBOT",
        },
    },
    "FIREWORKS": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "default_model": os.getenv("MODEL_NAME", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
        "key_env": "FIREWORKS_API_KEY",
        "secret_key": "FIREWORKS_API_KEY",
        "extra_headers": {},
    },
    "DEEPINFRA": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "default_model": os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        "key_env": "DEEPINFRA_API_KEY",
        "secret_key": "DEEPINFRA_API_KEY",
        "extra_headers": {},
    },
}

cfg = PROVIDERS.get(PROVIDER, PROVIDERS["GROQ"])

# ---------------- Embeddings + KB/FAQ config ----------------
EMBEDDING_MODELS = {
    "GROQ": "text-embedding-3-small",
    "TOGETHER": "togethercomputer/m2-bert-80M-8k-retrieval",
    "OPENROUTER": "openrouter/clip-vit-large-patch14",
    "FIREWORKS": "nomic-ai/nomic-embed-text-v1",
    "DEEPINFRA": "text-embedding-3-small",
}
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODELS.get(PROVIDER, "text-embedding-3-small"))

KB_DIR = os.getenv("KB_DIR", "kb")
CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters
TOP_K = 4
USE_KB_DEFAULT = True  # sidebar toggle default

# FAQ matching thresholds
FAQ_MIN_COSINE = 0.80          # use FAQ answer if cosine >= this (when embeddings available)
FAQ_MIN_KEYWORD_OVERLAP = 0.55 # use FAQ answer if keyword overlap >= this (fallback)
FAQ_TOP_K = 1

# ---------------- KB data structures & helpers ----------------
@dataclass
class KBChunk:
    doc_path: str
    chunk_id: str
    text: str

def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if end == n:
            break
    return [c for c in chunks if c]

def load_kb_files(kb_dir: str = KB_DIR):
    paths = sorted(glob.glob(str(pathlib.Path(kb_dir) / "**/*"), recursive=True))
    docs = []
    for p in paths:
        if os.path.isdir(p):
            continue
        if not any(p.lower().endswith(ext) for ext in (".md", ".txt")):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                docs.append((p, f.read()))
        except Exception:
            continue
    return docs

def build_kb_chunks():
    docs = load_kb_files()
    chunks: list[KBChunk] = []
    for path, content in docs:
        for i, c in enumerate(chunk_text(content)):
            chunks.append(KBChunk(doc_path=path, chunk_id=f"{_hash(path)}-{i}", text=c))
    return chunks

def cosine(a, b):
    import math
    if not a or not b:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def keyword_score(q: str, t: str) -> float:
    """
    Simple TF-like overlap (unbounded). Kept for KB ranking.
    """
    q_tokens = [w for w in re.findall(r"\b\w+\b", q.lower()) if len(w) > 2]
    t_tokens = re.findall(r"\b\w+\b", t.lower())
    if not q_tokens or not t_tokens:
        return 0.0
    hits = sum(t_tokens.count(w) for w in set(q_tokens))
    return hits / (len(set(q_tokens)) + 1e-6)

def keyword_overlap_01(a: str, b: str) -> float:
    """
    Normalized token overlap in [0,1] for FAQ matching (safer thresholding).
    """
    at = set(w for w in re.findall(r"\b\w+\b", a.lower()) if len(w) > 2)
    bt = set(w for w in re.findall(r"\b\w+\b", b.lower()) if len(w) > 2)
    if not at or not bt:
        return 0.0
    inter = len(at & bt)
    return inter / max(1, len(at))

# ---------------- Client bootstrap ----------------
def get_api_key():
    k = os.getenv(cfg["key_env"])
    if k:
        return k
    try:
        return st.secrets[cfg["secret_key"]]
    except Exception:
        return None

API_KEY = get_api_key()

st.set_page_config(page_title="ESCP ChatBOT", page_icon="★")
st.title("ESCP ChatBOT")

if not API_KEY:
    st.error(
        f"Missing API key for {PROVIDER}.\n"
        f"Set **{cfg['key_env']}** in Streamlit → Settings → Secrets, "
        f"or as an environment variable."
    )
    with st.sidebar.expander("Debug config"):
        st.write({"PROVIDER": PROVIDER, "needed_key_name": cfg["key_env"]})
    st.stop()

client = OpenAI(api_key=API_KEY, base_url=cfg["base_url"])  # OpenAI-compatible client
EXTRA_HEADERS = cfg["extra_headers"]

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Try to call an OpenAI-compatible embeddings endpoint via the selected provider.
    If that fails, raise to trigger keyword fallback.
    """
    try:
        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            extra_headers=EXTRA_HEADERS or None,
        )
        return [d.embedding for d in emb.data]
    except Exception as e:
        raise RuntimeError(f"Embeddings unavailable on {PROVIDER} ({EMBED_MODEL}): {e}")

# ---------------- KB index ----------------
@st.cache_resource(show_spinner=False)
def get_kb_index():
    chunks = build_kb_chunks()
    index = {"embeddings": None, "chunks": chunks, "provider": PROVIDER, "model": EMBED_MODEL}
    try:
        texts = [c.text for c in chunks]
        if texts:
            embs = embed_texts(texts)
            index["embeddings"] = embs
    except Exception:
        index["embeddings"] = None  # fallback will be keyword scoring
    return index

def retrieve_kb(query: str, k: int = TOP_K):
    idx = get_kb_index()
    chunks = idx["chunks"]
    if not chunks:
        return []
    # Embedding path
    if idx["embeddings"] is not None:
        try:
            qv = embed_texts([query])[0]
            scored = []
            for c, ev in zip(chunks, idx["embeddings"]):
                scored.append((cosine(qv, ev), c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:k]
        except Exception:
            pass
    # Fallback keyword path
    scored = [(keyword_score(query, c.text), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

# ---------------- FAQ index (questions only) ----------------
def load_faqs(path: str = "faqs.json") -> List[Dict[str, str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect objects with: theme, q, a
        cleaned = []
        for it in data:
            q = (it.get("q") or "").strip()
            a = (it.get("a") or "").strip()
            theme = (it.get("theme") or "FAQ").strip()
            if q and a:
                cleaned.append({"q": q, "a": a, "theme": theme})
        return cleaned
    except FileNotFoundError:
        return []
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def get_faq_index():
    items = load_faqs()
    index = {"questions": [it["q"] for it in items], "answers": [it["a"] for it in items],
             "themes": [it["theme"] for it in items], "embeddings": None}
    if not items:
        return index
    # Try to embed questions
    try:
        q_embs = embed_texts(index["questions"])
        index["embeddings"] = q_embs
    except Exception:
        index["embeddings"] = None
    return index

def match_faq(query: str) -> Optional[Tuple[str, str, float]]:
    """
    Try to match a user query to a single FAQ answer.
    Returns (answer, question, score) or None.
    """
    idx = get_faq_index()
    questions = idx["questions"]
    answers = idx["answers"]
    themes = idx["themes"]

    if not questions:
        return None

    # Embedding path (cosine)
    if idx["embeddings"] is not None:
        try:
            qv = embed_texts([query])[0]
            scores = [(cosine(qv, ev), i) for i, ev in enumerate(idx["embeddings"])]
            scores.sort(reverse=True, key=lambda x: x[0])
            top_score, top_i = scores[0]
            if top_score >= FAQ_MIN_COSINE:
                return answers[top_i], questions[top_i], float(top_score)
        except Exception:
            pass

    # Fallback keyword overlap in [0,1]
    best_i = -1
    best_s = -1.0
    for i, q in enumerate(questions):
        s = keyword_overlap_01(query, q)
        if s > best_s:
            best_s, best_i = s, i
    if best_i >= 0 and best_s >= FAQ_MIN_KEYWORD_OVERLAP:
        return answers[best_i], questions[best_i], float(best_s)

    return None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Settings")
    model = st.text_input(
        "Model",
        value=cfg["default_model"],
        help="Override default model if you like."
    )
    system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=80)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 300, 16)
    use_stream = st.toggle("Stream responses", value=True)
    st.caption("If responses stop early, increase Max new tokens.")

    st.divider()
    st.subheader("Knowledge Base")
    use_kb = st.toggle("Use KB for answers", value=USE_KB_DEFAULT, help="Retrieve relevant KB chunks and inject them into the system prompt.")
    colA, colB = st.columns(2)
    if colA.button("Rebuild KB Index"):
        get_kb_index.clear()
        get_kb_index()
        st.success("KB index rebuilt.")
    if colB.button("Reload FAQs"):
        get_faq_index.clear()
        get_faq_index()
        st.success("FAQs reloaded.")

    kb_files = load_kb_files()
    st.caption(f"{len(kb_files)} KB documents found in `{KB_DIR}/`")

    st.divider()
    HISTORY_FILE = "history.json"
    col1, col2, col3 = st.columns(3)
    if col1.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()
    if col2.button("Save chat"):
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.get("messages", []), f, ensure_ascii=False, indent=2)
            st.success(f"Saved to {HISTORY_FILE}")
        except Exception as e:
            st.error(f"Save failed: {e}")
    if col3.button("Load chat"):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
            st.success(f"Loaded from {HISTORY_FILE}")
            st.rerun()
        except FileNotFoundError:
            st.warning("No history file found.")
        except Exception as e:
            st.error(f"Load failed: {e}")

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# ---------------- OpenAI-style chat call ----------------
def chat_generate(messages: List[Dict[str, str]]) -> str:
    """
    Calls provider via OpenAI-compatible Chat Completions.
    messages: [{"role":"system/user/assistant","content":"..."}]
    """
    content = ""
    try:
        if use_stream:
            with st.spinner("Thinking…"):
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    stream=True,
                    extra_headers=EXTRA_HEADERS or None,
                )
                placeholder = st.empty()
                for chunk in stream:
                    delta = (chunk.choices[0].delta.content or "")
                    if delta:
                        content += delta
                        placeholder.markdown(content)
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_headers=EXTRA_HEADERS or None,
            )
            content = resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Generation failed: {e}")
    return content.strip()

# ---------------- UI: render history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Handle input ----------------
if prompt := st.chat_input("Type your message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 1) Try FAQ first (exact canned answer if strong match) ---
    faq_hit = match_faq(prompt)
    if faq_hit:
        answer, matched_q, score = faq_hit
        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"Matched FAQ: “{matched_q}”  •  score={score:.2f}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # --- 2) Otherwise, compose messages (system + history), with optional KB context ---
        final_system_prompt = system_prompt or "You are a helpful assistant."
        if use_kb:
            hits = retrieve_kb(prompt, k=TOP_K)
            if hits:
                sources_md = []
                ctx_snippets = []
                for score, chunk in hits:
                    rel = f"{score:.3f}"
                    sources_md.append(f"- `{os.path.basename(chunk.doc_path)}` (score {rel})")
                    snippet = chunk.text.strip()
                    if len(snippet) > 1200:
                        snippet = snippet[:1200] + "…"
                    ctx_snippets.append(f"[SOURCE: {os.path.basename(chunk.doc_path)}]\n{snippet}")
                kb_context = "\n\n".join(ctx_snippets)
                kb_context_blocks = [
                    "You are a helpful assistant. Use the following knowledge base excerpts as trusted context. If the user asks about ESCP internships/PECS or related processes, rely on these. If uncertain, say so briefly.",
                    "\n---\nKB CONTEXT START\n",
                    kb_context,
                    "\nKB CONTEXT END\n---\n",
                    "When answering, cite the filename(s) you used in square brackets, e.g., [internship_policy.md]."
                ]
                final_system_prompt = (system_prompt or "You are a helpful assistant.") + "\n\n" + "\n".join(kb_context_blocks)

        msgs = [{"role": "system", "content": final_system_prompt}]
        msgs.extend(st.session_state.messages)

        with st.chat_message("assistant"):
            reply = chat_generate(msgs)
            st.markdown(reply or "_(no text returned)_")

            # Optional: show “Sources used” for transparency when KB is on
            if use_kb:
                hits_for_display = retrieve_kb(prompt, k=TOP_K)
                if hits_for_display:
                    with st.expander("Sources used"):
                        for score, chunk in hits_for_display:
                            st.write(f"• {os.path.basename(chunk.doc_path)}  (score {score:.3f})")

        st.session_state.messages.append({"role": "assistant", "content": reply if reply else ""})
