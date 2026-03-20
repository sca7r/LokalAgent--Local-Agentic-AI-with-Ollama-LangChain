from langchain.memory import ConversationBufferWindowMemory


def get_memory(window_size: int = 10):
    """
    Returns a sliding-window conversation memory.
    Keeps the last `window_size` exchanges in context —
    enough history to be useful without blowing the context window.
    """
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=window_size,
        return_messages=True,
    )