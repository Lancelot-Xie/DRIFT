#!/usr/bin/env python3
"""
DRIFT Demo - Gradio-based chat interface with persistent history.

This demo provides a chat interface for the DRIFT framework with
task history that persists across browser refreshes.
"""

import json
import os
import time
import uuid

import gradio as gr

# Directory for storing history files
HISTORY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "task_history.json")


def ensure_history_dir():
    """Ensure the history directory exists."""
    os.makedirs(HISTORY_DIR, exist_ok=True)


def load_history():
    """Load complete task history from disk.

    This function reads the history file every time it is called,
    ensuring that the latest history is always returned (e.g. on
    browser refresh).
    """
    ensure_history_dir()
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_history(history):
    """Save the full task history to disk."""
    ensure_history_dir()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def format_history_list(history):
    """Format task history into a display string for the sidebar."""
    if not history:
        return "No task history yet."
    lines = []
    for i, task in enumerate(history, 1):
        timestamp = task.get("timestamp", "unknown")
        question = task.get("question", "")
        # Truncate long questions for display
        if len(question) > 60:
            question = question[:60] + "..."
        lines.append(f"{i}. [{timestamp}] {question}")
    return "\n".join(lines)


def chat_respond(message, chat_history):
    """Process a user message and return a response.

    In a real deployment this would call the DRIFT inference pipeline.
    For the demo, a placeholder response is returned.
    """
    if not message or not message.strip():
        return "", chat_history, format_history_list(load_history())

    # Placeholder response (replace with actual DRIFT inference)
    response = (
        f"[DRIFT Demo] Received your question: {message}\n\n"
        "This is a placeholder response. In production, this would be "
        "processed by the DRIFT dual-model framework for efficient "
        "long-context inference."
    )

    # Update chat history displayed in the chat window
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    # Save to persistent task history
    task_history = load_history()
    task_history.append({
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": message,
        "response": response,
    })
    save_history(task_history)

    return "", chat_history, format_history_list(load_history())


def load_on_refresh():
    """Callback invoked every time the page loads or is refreshed.

    This is the key fix: instead of loading history only at server
    startup, we reload it from disk on every page load so that the
    UI always reflects the latest persisted state.
    """
    history = load_history()
    history_display = format_history_list(history)
    return [], history_display


# --------------- Build Gradio UI ---------------

with gr.Blocks(title="DRIFT Demo") as demo:
    gr.Markdown("# 🚀 DRIFT Demo\nDecoupled Reasoning with Implicit Fact Tokens")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat")
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your question",
                    placeholder="Type your question here...",
                    scale=4,
                )
                send_btn = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            history_display = gr.Textbox(
                label="Task History",
                interactive=False,
                lines=20,
            )

    # Send message on button click or Enter key
    send_btn.click(
        chat_respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, history_display],
    )
    msg_input.submit(
        chat_respond,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, history_display],
    )

    # ---- KEY FIX ----
    # Reload history from disk every time the page loads or is refreshed.
    # This ensures that new history entries saved during the current session
    # (or by other sessions) are visible after a browser refresh, rather
    # than showing only the history that was present when the server started.
    demo.load(
        load_on_refresh,
        inputs=None,
        outputs=[chatbot, history_display],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
