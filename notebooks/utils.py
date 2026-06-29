from contextlib import asynccontextmanager
import json
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def patch_mcp_stdio_for_jupyter() -> None:
    """Patch MCP stdio transport for Jupyter notebooks on Windows.

    Jupyter replaces sys.stderr with streams that lack fileno(), which breaks
    MCP stdio subprocess creation when asyncio falls back to subprocess.Popen.
    Also patches Windows process helpers so stderr is redirected safely.
    """
    import io

    if sys.platform == "win32":
        import asyncio

        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    def _normalize_errlog(errlog):
        if errlog is subprocess.PIPE or errlog is subprocess.DEVNULL:
            return errlog
        if errlog is None:
            return subprocess.PIPE
        try:
            errlog.fileno()
            return errlog
        except (AttributeError, io.UnsupportedOperation, ValueError, OSError):
            return subprocess.PIPE

    if sys.platform == "win32":
        from mcp.os.win32 import utilities as win32_util

        if not getattr(win32_util, "_jupyter_stdio_patched", False):
            _orig_create = win32_util.create_windows_process
            _orig_fallback = win32_util._create_windows_fallback_process

            async def _patched_create_windows_process(
                command, args, env=None, errlog=sys.stderr, cwd=None
            ):
                return await _orig_create(
                    command, args, env, _normalize_errlog(errlog), cwd
                )

            async def _patched_fallback_process(
                command, args, env=None, errlog=sys.stderr, cwd=None
            ):
                return await _orig_fallback(
                    command, args, env, _normalize_errlog(errlog), cwd
                )

            win32_util.create_windows_process = _patched_create_windows_process
            win32_util._create_windows_fallback_process = _patched_fallback_process
            win32_util._jupyter_stdio_patched = True

    import mcp.client.stdio as mcp_stdio

    if not getattr(mcp_stdio, "_jupyter_stdio_patched", False):
        _original_stdio_client = mcp_stdio.stdio_client

        @asynccontextmanager
        async def jupyter_stdio_client(server, errlog=subprocess.PIPE):
            async with _original_stdio_client(
                server, errlog=_normalize_errlog(errlog)
            ) as streams:
                yield streams

        mcp_stdio.stdio_client = jupyter_stdio_client
        mcp_stdio._jupyter_stdio_patched = True
        mcp_stdio._jupyter_stdio_client = jupyter_stdio_client

    jupyter_stdio_client = mcp_stdio._jupyter_stdio_client

    try:
        import langchain_mcp_adapters.sessions as lcm_sessions

        lcm_sessions.stdio_client = jupyter_stdio_client
    except ImportError:
        pass

def format_message_content(message):
    """Convert message content to displayable string"""
    parts = []
    tool_calls_processed = False
    
    # Handle main content
    if isinstance(message.content, str):
        parts.append(message.content)
    elif isinstance(message.content, list):
        # Handle complex content like tool calls (Anthropic format)
        for item in message.content:
            if item.get('type') == 'text':
                parts.append(item['text'])
            elif item.get('type') == 'tool_use':
                parts.append(f"\n🔧 Tool Call: {item['name']}")
                parts.append(f"   Args: {json.dumps(item['input'], indent=2)}")
                parts.append(f"   ID: {item.get('id', 'N/A')}")
                tool_calls_processed = True
    else:
        parts.append(str(message.content))
    
    # Handle tool calls attached to the message (OpenAI format) - only if not already processed
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


def format_message(messages):
    """Alias for format_messages for backward compatibility"""
    return format_messages(messages)


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
    formatted_text.highlight_regex(r'<[^>]+>', style="bold blue")  # Highlight XML tags
    formatted_text.highlight_regex(r'##[^#\n]+', style="bold magenta")  # Highlight headers
    formatted_text.highlight_regex(r'###[^#\n]+', style="bold cyan")  # Highlight sub-headers

    # Display in a panel for better presentation
    console.print(Panel(
        formatted_text, 
        title=f"[bold green]{title}[/bold green]",
        border_style=border_style,
        padding=(1, 2)
    ))