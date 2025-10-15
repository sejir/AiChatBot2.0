import os
import json
from typing import List, Dict

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

# OpenAI-compatible client
client = OpenAI(api_key=API_KEY, base_url=cfg["base_url"])
EXTRA_HEADERS = cfg["extra_headers"]

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
                    delta = chunk.choices[0].delta.content or ""
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

    # Compose messages (system + history)
    msgs = [{"role": "system", "content": system_prompt}]
    msgs.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        reply = chat_generate(msgs)
        st.markdown(reply or "_(no text returned)_")

    st.session_state.messages.append({"role": "assistant", "content": reply})
