"""
Task 1: Streamlit Interface for a Locally Installed LLM (via Ollama)
=====================================================================
Requirements:
  - pip install streamlit requests
  - Install Ollama: https://ollama.com
  - Pull a model: ollama pull llama3
Run:
  streamlit run task1_ollama_streamlit_app.py
"""

import streamlit as st
import requests
import json

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "llama3"          # change to any model you have pulled


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def get_available_models():
    """Fetch all models currently pulled in Ollama."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except requests.exceptions.ConnectionError:
        return []
    except Exception:
        return []


def stream_ollama_response(model: str, messages: list):
    """
    Call Ollama's /api/chat endpoint with streaming.
    Yields text chunks as they arrive.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
    }
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"].get("content", "")
                    if chunk.get("done", False):
                        break
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️  **Cannot connect to Ollama.** Make sure Ollama is running (`ollama serve`) and a model is pulled (`ollama pull llama3`)."
    except Exception as e:
        yield f"\n\n⚠️  **Error:** {e}"


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Local LLM Chat",
    page_icon="🤖",
    layout="wide",
)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    # Model selector
    available_models = get_available_models()
    if available_models:
        selected_model = st.selectbox(
            "🧠 Select Model",
            options=available_models,
            index=0,
            help="Models pulled via `ollama pull <name>`",
        )
    else:
        st.error("⚠️ Ollama not reachable.\nStart it with:\n```\nollama serve\n```")
        selected_model = st.text_input(
            "Model name (manual)",
            value=DEFAULT_MODEL,
            help="Type the model name you have pulled.",
        )

    st.markdown("---")

    # System prompt
    system_prompt = st.text_area(
        "🗒️ System Prompt",
        value="You are a helpful, concise, and friendly AI assistant.",
        height=120,
        help="Sets the behaviour of the model throughout the conversation.",
    )

    st.markdown("---")

    # Temperature slider
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative; Lower = more deterministic.",
    )

    st.markdown("---")

    # Reset button
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.token_count = 0
        st.rerun()

    # Stats
    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    msg_count = len([m for m in st.session_state.get("messages", []) if m["role"] == "user"])
    st.metric("Messages sent", msg_count)

    st.markdown("---")
    st.caption("Powered by [Ollama](https://ollama.com) + Streamlit")


# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": ..., "content": ...}


# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────
st.title("🤖 Local LLM Chat")
st.caption(f"Talking to: **{selected_model}**  |  Temperature: **{temperature}**")
st.markdown("---")

# ── Conversation history panel ────────────────
history_container = st.container()
with history_container:
    if not st.session_state.messages:
        st.info("👋 Start a conversation below! Your chat history will appear here.")
    else:
        for msg in st.session_state.messages:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

# ── Input box ────────────────────────────────
user_input = st.chat_input("Type your message here…")

if user_input:
    # 1. Display user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # 2. Build message list for Ollama (include system prompt)
    ollama_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    # 3. Stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in stream_ollama_response(selected_model, ollama_messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")   # blinking cursor effect
        response_placeholder.markdown(full_response)

    # 4. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "💡 **Tip:** Pull models with `ollama pull llama3` | `ollama pull deepseek-r1` | `ollama pull mistral`"
)
