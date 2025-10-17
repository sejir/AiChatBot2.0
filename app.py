import os
import json
from typing import List, Dict
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
        # see https://console.groq.com for latest models
        "default_model": os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        "key_env": "GROQ_API_KEY",
        "secret_key": "GROQ_API_KEY",
        "extra_headers": {},  # none needed
    },
    "TOGETHER": {
        "base_url": "https://api.together.xyz/v1",
        # good choices: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo, mistralai/Mixtral-8x7B-Instruct-v0.1
        "default_model": os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "key_env": "TOGETHER_API_KEY",
        "secret_key": "TOGETHER_API_KEY",
        "extra_headers": {},
    },
    "OPENROUTER": {
        "base_url": "https://openrouter.ai/api/v1",
        # many free/cheap models; e.g. meta-llama/llama-3.1-8b-instruct:free (availability varies)
        "default_model": os.getenv("MODEL_NAME", "meta-llama/llama-3.1-8b-instruct"),
        "key_env": "OPENROUTER_API_KEY",
        "secret_key": "OPENROUTER_API_KEY",
        "extra_headers": {
            # Optional but recommended by OpenRouter (if you have a site):
            # "HTTP-Referer": "https://your-app-url",
            # "X-Title": "ESCP ChatBOT",
        },
    },
    "FIREWORKS": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        # example: accounts/fireworks/models/llama-v3p1-8b-instruct
        "default_model": os.getenv("MODEL_NAME", "accounts/fireworks/models/llama-v3p1-8b-instruct"),
        "key_env": "FIREWORKS_API_KEY",
        "secret_key": "FIREWORKS_API_KEY",
        "extra_headers": {},
    },
    "DEEPINFRA": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        # e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
        "default_model": os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        "key_env": "DEEPINFRA_API_KEY",
        "secret_key": "DEEPINFRA_API_KEY",
        "extra_headers": {},
    },
}

cfg = PROVIDERS.get(PROVIDER, PROVIDERS["GROQ"])

# ---------------- Embeddings + KB config ----------------
EMBEDDING_MODELS = {
    "GROQ": "text-embedding-3-small",
    "TOGETHER": "togethercomputer/m2-bert-80M-8k-retrieval",
    "OPENROUTER": "openrouter/clip-vit-large-patch14",
    "FIREWORKS": "nomic-ai/nomic-embed-text-v1",
    "DEEPINFRA": "text-embedding-3-small"
}
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODELS.get(PROVIDER, "text-embedding-3-small"))
KB_DIR = os.getenv("KB_DIR", "kb")
KB_INDEX = os.getenv("KB_INDEX", "kb_index.json")  # reserved if you want to persist to disk later

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters
TOP_K = 4
USE_KB_DEFAULT = True  # sidebar toggle default

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
    q_tokens = [w for w in re.findall(r"\b\w+\b", q.lower()) if len(w) > 2]
    t_tokens = re.findall(r"\b\w+\b", t.lower())
    if not q_tokens or not t_tokens:
        return 0.0
    hits = sum(t_tokens.count(w) for w in set(q_tokens))
    return hits / (len(set(q_tokens)) + 1e-6)

# ---------------- Client bootstrap ----------------
def get_api_key():
    # 1) env var (works locally & on Streamlit Cloud Environment Variables)
    k = os.getenv(cfg["key_env"])
    if k:
        return k
    # 2) Streamlit Secrets (Settings → Secrets on Streamlit Cloud)
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

def embed_texts(texts: list[str]) -> list[list[float]]:
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

def retrieve(query: str, k: int = TOP_K):
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

    kb_files = load_kb_files()
    st.caption(f"{len(kb_files)} KB documents found in `{KB_DIR}/`")
    if st.checkbox("Preview KB files", value=False):
        for p, _ in kb_files[:20]:
            st.write("•", p)

    st.divider()
    st.subheader("FAQs")
    try:
        with open("faqs.json", "r", encoding="utf-8") as f:
            faqs = json.load(f)
        for item in faqs:
            with st.expander(item.get("q", "FAQ")):
                st.markdown(item.get("a", ""))
    except FileNotFoundError:
        st.info("Create a `faqs.json` file to show FAQs here.")
    except Exception as e:
        st.error(f"Failed to load FAQs: {e}")

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

    # --- Compose messages (system + history), with optional KB context ---
    final_system_prompt = system_prompt or "You are a helpful assistant."
    if use_kb and prompt:
        hits = retrieve(prompt, k=TOP_K)
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
        if use_kb and prompt:
            hits_for_display = retrieve(prompt, k=TOP_K)
            if hits_for_display:
                with st.expander("Sources used"):
                    for score, chunk in hits_for_display:
                        st.write(f"• {os.path.basename(chunk.doc_path)}  (score {score:.3f})")

    st.session_state.messages.append({"role": "assistant", "content": reply})
