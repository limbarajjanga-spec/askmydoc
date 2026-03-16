# llm/claude_client.py
import anthropic
from config.settings import ANTHROPIC_API_KEY
from config.constants import CLAUDE_MODEL, CLAUDE_MAX_TOKENS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a helpful document assistant.
Answer questions based ONLY on the document context provided.
Each context chunk includes a page number — always cite it like (Page 3).

Rules:
1. Answer using ONLY the provided context
2. Always mention the page number where you found the answer
3. If the answer is not in the context say:
   "I couldn't find this information in the uploaded document."
4. Never make up facts or guess
5. You have access to the previous conversation history —
   use it to understand follow-up questions"""


def ask_claude(question: str,
               context_chunks: list[dict],
               chat_history: list[dict] = None) -> str:
    """
    Args:
        question:      user's question
        context_chunks: retrieved chunks with page metadata
        chat_history:  list of previous messages
                       [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]
    """
    # Build context with page labels
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[Page {chunk['page']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Current question with context
    current_message = f"""Here is the relevant context from the document:

{context}

---

Based on the context above, answer this question:
{question}"""

    # Build full message history for Claude
    # Previous turns + current question
    messages = []

    if chat_history:
        for msg in chat_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Add current question as last message
    messages.append({"role": "user", "content": current_message})

    print(f"[claude] Sending question with {len(messages)} messages in history...")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages
    )

    answer = response.content[0].text
    print(f"[claude] Done ({response.usage.input_tokens} in / "
          f"{response.usage.output_tokens} out tokens)")
    return answer