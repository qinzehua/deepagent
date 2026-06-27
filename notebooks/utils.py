from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import json

console = Console()

def format_message_content(message):
    """Convert message content to displayable string.

    The incoming `message` is usually a LangChain message object such as
    HumanMessage, AIMessage, or ToolMessage. Its shape can vary:
    - `message.content` may be a plain string, e.g. "Hello"
    - `message.content` may be a list of content blocks, e.g.
      [{"type": "text", "text": "Hello"}]
    - `message.content` may contain tool-use blocks in Anthropic style, e.g.
      [{"type": "tool_use", "name": "search", "input": {...}, "id": "call_1"}]
    - `message.tool_calls` may exist separately for OpenAI-style messages,
      e.g. [{"name": "search", "args": {...}, "id": "call_1"}]
    - `message.content` may also be some other non-string/non-list value,
      in which case we fall back to `str(message.content)`.
    """
    parts = []
    tool_calls_processed = False

    # Handle the main content payload. This is often a plain string.
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like content blocks or tool-use blocks.
        # Typical examples:
        #   [{"type": "text", "text": "Hello"}]
        #   [{"type": "tool_use", "name": "search", "input": {"query": "foo"}, "id": "call_1"}]
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        # Fallback for unexpected content shapes.
        parts.append(str(message.content))

    # Handle tool calls attached to the message in OpenAI-style format.
    # This is separate from `message.content` and is only used when the
    # content blocks above did not already process tool-use information.
    if not tool_calls_processed and hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            parts.append(f"\n🔧 Tool Call: {tool_call['name']}")
            parts.append(f"   Args: {json.dumps(tool_call['args'], indent=2)}")
            parts.append(f"   ID: {tool_call['id']}")

    return "\n".join(parts)

def format_messages(messages):
    """Format and display a list of messages with Rich formatting"""
    for m in messages:
        msg_type = m.__class__.__name__.replace('Message', '')
        content = format_message_content(m)

        if msg_type == 'Human':
            console.print(Panel(content, title="🧑 Human", border_style="blue"))
        elif msg_type == 'Ai':
            console.print(Panel(content, title="🤖 Assistant", border_style="green"))
        elif msg_type == 'Tool':
            console.print(Panel(content, title="🔧 Tool Output", border_style="yellow"))
        else:
            console.print(Panel(content, title=f"📝 {msg_type}", border_style="white"))

def show_prompt(prompt_text: str, title: str = "Prompt", border_style: str = "blue"):
    """
    Display a prompt with rich formatting and XML tag highlighting.

    Args:
        prompt_text: The prompt string to display
        title: Title for the panel (default: "Prompt")
        border_style: Border color style (default: "blue")
    """
    # Create a formatted display of the prompt
    formatted_text = Text(prompt_text)
    formatted_text.highlight_regex(r'<[^>]+>', style="bold yellow")  # Highlight XML tags
    formatted_text.highlight_regex(r'##[^#\n]+', style="bold yellow")  # Highlight headers
    formatted_text.highlight_regex(r'###[^#\n]+', style="bold cyan")  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(Panel(
        formatted_text,
        title=f"[bold green]{title}[/bold green]",
        border_style=border_style,
        padding=(1, 2)
    ))