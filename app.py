import os
import json
from typing import List, Dict

import streamlit as st
from huggingface_hub import InferenceClient

# ---------------- Config ----------------
import os, streamlit as st

def get_hf_token():
    
    # 1) Environment variable wins (works locally & on Streamlit Cloud)
    t = os.getenv("HF_TOKEN")
    if t:
        return t

    # 2) Try Streamlit secrets (only if configured)
    try:
        return st.secrets["HF_TOKEN"]
    except Exception:
        return None

HF_TOKEN = get_hf_token()

st.set_page_config(page_title="ESCP ChatBOT", page_icon="★")

if not HF_TOKEN:
    st.error(
        "HF_TOKEN not found.\n\n"
        "Set it EITHER as an Environment Variable (HF_TOKEN) OR in Streamlit Secrets (HF_TOKEN)."
    )
    # optional: quick diagnostics
    has_env = bool(os.getenv("HF_TOKEN"))
    has_secrets = False
    try:
        _ = st.secrets["HF_TOKEN"]
        has_secrets = True
    except Exception:
        pass
    with st.sidebar.expander("Debug config"):
        st.write({"env_HF_TOKEN": has_env, "secrets_HF_TOKEN": has_secrets})
    st.stop()


client = InferenceClient(provider="featherless-ai", api_key=HF_TOKEN)
# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.subheader("Settings")
    model = st.text_input("Model", value=DEFAULT_MODEL, help="e.g., meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-Instruct")
    system_prompt = st.text_area("System prompt", value="You are a helpful assistant.", height=80)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 16, 1024, 300, 16)
    use_stream = st.toggle("Stream responses", value=True)
    st.caption("If responses stop early, increase Max new tokens.")

    st.divider()
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

# ---------------- Client ----------------
try:
    client = InferenceClient(
        provider="featherless-ai",
        api_key=HF_TOKEN,
    )
except Exception as e:
    st.error(f"Failed to init HF client: {e}")
    st.stop()

# ---------------- State ----------------
if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# ---------------- Prompt builder (Llama-friendly) ----------------
# --- Build a clean prompt (system + short history) ---
def build_llama_prompt(history, system: str = "You are a helpful assistant.", max_turns: int = 4) -> str:
    """
    Formats as:
      system: ...
      user: ...
      assistant: ...
    Ends with 'assistant:' so the model completes that turn.
    Keeps only the last few turns to avoid runaway patterns.
    """
    # keep last `max_turns` messages
    trimmed = history[-(max_turns*2):] if max_turns else history
    lines = [f"system: {system.strip()}"]
    for m in trimmed:
        role = "assistant" if m["role"] == "assistant" else "user"
        lines.append(f"{role}: {m['content']}")
    lines.append("assistant:")
    return "\n".join(lines)


# --- Generate with stop sequences (prevents echoing 'user:' / 'assistant:') ---
def generate_reply(prompt_text: str, *, stream: bool, max_tokens: int, temp: float) -> str:
    stop_sequences = ["\nuser:", "\nassistant:", "\nsystem:"]  # <- key change
    partial = ""
    debug_chunks = []

    if stream:
        placeholder = st.empty()
        try:
            for chunk in client.text_generation(
                prompt_text,
                model=model,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=0.95,
                repetition_penalty=1.05,
                stream=True,
                details=True,
                return_full_text=False,
                stop_sequences=stop_sequences,  # <- key change
            ):
                debug_chunks.append(repr(chunk))
                piece = ""
                if isinstance(chunk, str):
                    piece = chunk
                else:
                    token_obj = getattr(chunk, "token", None)
                    piece = getattr(token_obj, "text", None)
                    if not piece and isinstance(chunk, dict):
                        piece = (
                            (chunk.get("token") or {}).get("text")
                            or chunk.get("generated_text")
                            or ""
                        )
                if not piece:
                    continue
                partial += piece
                placeholder.markdown(partial)
        except Exception as e:
            st.warning(f"Stream error, retrying non-stream: {e}")

    if not partial:
        try:
            out = client.text_generation(
                prompt_text,
                model=model,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=0.95,
                repetition_penalty=1.05,
                stream=False,
                details=True,
                return_full_text=False,
                stop_sequences=stop_sequences,  # <- key change
            )
            partial = out if isinstance(out, str) else (
                getattr(out, "generated_text", "") or (out.get("generated_text", "") if isinstance(out, dict) else "")
            )
        except Exception as e2:
            st.error(f"Generation failed: {e2}")
            partial = ""

    with st.sidebar.expander("Debug: last chunks", expanded=False):
        st.code("\n".join(debug_chunks[-10:]) or "(no chunks)")

    return (partial or "").strip()

# ---------------- UI: History ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Handle Input ----------------
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build prompt & generate
    full_prompt = build_llama_prompt(st.session_state.messages, system="You are a helpful assistant.")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_reply(full_prompt, stream=use_stream, max_tokens=max_new_tokens, temp=temperature)
            st.markdown(reply or "_(no text returned)_")

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
