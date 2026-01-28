"""Built-in tools for the agent framework.

This module exports all built-in tools that come with the framework,
including file, git, shell, search, and web tools.
"""

from .file_tools import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GlobTool,
    ListDirectoryTool,
    get_file_tools,
)

from .git_tools import (
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitBranchTool,
    GitLogTool,
    GitAddTool,
    get_git_tools,
    run_git_command,
)

from .shell_tools import (
    BashTool,
    BackgroundProcessTool,
    WorkingDirectoryTool,
    SandboxConfig,
    DANGEROUS_COMMANDS,
    DANGEROUS_PATTERNS,
    get_shell_tools,
)

from .search_tools import (
    GrepTool,
    RipgrepTool,
    SemanticSearchTool,
    SearchMatch,
    EmbeddingFunc,
    get_search_tools,
)

from .web_tools import (
    WebFetchTool,
    WebSearchTool,
    URLParserTool,
    FetchConfig,
    SearchConfig,
    SimpleHTMLTextExtractor,
    get_web_tools,
)

__all__ = [
    # File tools
    "ReadFileTool",
    "WriteFileTool",
    "EditFileTool",
    "GlobTool",
    "ListDirectoryTool",
    "get_file_tools",

    # Git tools
    "GitStatusTool",
    "GitDiffTool",
    "GitCommitTool",
    "GitBranchTool",
    "GitLogTool",
    "GitAddTool",
    "get_git_tools",
    "run_git_command",

    # Shell tools
    "BashTool",
    "BackgroundProcessTool",
    "WorkingDirectoryTool",
    "SandboxConfig",
    "DANGEROUS_COMMANDS",
    "DANGEROUS_PATTERNS",
    "get_shell_tools",

    # Search tools
    "GrepTool",
    "RipgrepTool",
    "SemanticSearchTool",
    "SearchMatch",
    "EmbeddingFunc",
    "get_search_tools",

    # Web tools
    "WebFetchTool",
    "WebSearchTool",
    "URLParserTool",
    "FetchConfig",
    "SearchConfig",
    "SimpleHTMLTextExtractor",
    "get_web_tools",
]


def get_all_builtin_tools(
    base_path=None,
    sandbox_config=None,
    fetch_config=None,
    search_config=None,
    embedding_func=None,
):
    """Get all built-in tools with common configuration.

    Args:
        base_path: Base path for file operations
        sandbox_config: Configuration for shell sandboxing
        fetch_config: Configuration for web fetching
        search_config: Configuration for web search
        embedding_func: Embedding function for semantic search

    Returns:
        List of all built-in tool instances
    """
    from pathlib import Path

    bp = Path(base_path) if base_path else None

    tools = []
    tools.extend(get_file_tools(base_path=bp))
    tools.extend(get_git_tools())
    tools.extend(get_shell_tools(sandbox=sandbox_config, working_dir=bp))
    tools.extend(get_search_tools(base_path=bp, embedding_func=embedding_func))
    tools.extend(get_web_tools(fetch_config=fetch_config, search_config=search_config))

    return tools
