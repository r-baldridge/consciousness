"""Web tools for fetching and searching web content.

This module provides tools for interacting with web resources,
including fetching pages and performing web searches via APIs.
"""

from __future__ import annotations
import asyncio
import json
import os
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse, quote_plus
from dataclasses import dataclass, field
from html.parser import HTMLParser
import ssl

from ..tool_base import Tool, ToolSchema, ToolResult, ToolPermission


# Check if aiohttp is available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class SimpleHTMLTextExtractor(HTMLParser):
    """Simple HTML parser that extracts text content."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self._skip_tags = {'script', 'style', 'noscript', 'meta', 'link'}
        self._current_tag = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple]):
        self._current_tag = tag.lower()
        if self._current_tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        if tag.lower() in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        self._current_tag = None

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return '\n'.join(self.text_parts)


@dataclass
class FetchConfig:
    """Configuration for web fetching.

    Attributes:
        timeout: Request timeout in seconds
        max_content_size: Maximum content size in bytes
        follow_redirects: Whether to follow redirects
        max_redirects: Maximum number of redirects to follow
        user_agent: User agent string to send
        verify_ssl: Whether to verify SSL certificates
        allowed_domains: List of allowed domains (None for all)
        blocked_domains: List of blocked domains
    """
    timeout: float = 30.0
    max_content_size: int = 10 * 1024 * 1024  # 10MB
    follow_redirects: bool = True
    max_redirects: int = 5
    user_agent: str = "AgentFramework/1.0"
    verify_ssl: bool = True
    allowed_domains: Optional[List[str]] = None
    blocked_domains: List[str] = field(default_factory=list)

    def is_domain_allowed(self, url: str) -> tuple[bool, Optional[str]]:
        """Check if a URL's domain is allowed.

        Args:
            url: The URL to check

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check blocked domains
            for blocked in self.blocked_domains:
                if domain == blocked or domain.endswith(f".{blocked}"):
                    return False, f"Domain '{domain}' is blocked"

            # Check allowed domains if specified
            if self.allowed_domains:
                for allowed in self.allowed_domains:
                    if domain == allowed or domain.endswith(f".{allowed}"):
                        return True, None
                return False, f"Domain '{domain}' is not in allowed list"

            return True, None

        except Exception as e:
            return False, f"Invalid URL: {e}"


class WebFetchTool(Tool):
    """Tool for fetching and parsing web pages."""

    def __init__(self, config: Optional[FetchConfig] = None):
        """Initialize the web fetch tool.

        Args:
            config: Fetch configuration
        """
        self._config = config or FetchConfig()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_fetch",
            description="Fetch content from a URL and optionally extract text.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to fetch"
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract plain text from HTML"
                },
                "headers": {
                    "type": "object",
                    "description": "Additional HTTP headers"
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds"
                }
            },
            required=["url"],
            permissions=[ToolPermission.NETWORK]
        )

    async def execute(
        self,
        url: str,
        extract_text: bool = True,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> ToolResult:
        """Fetch content from a URL.

        Args:
            url: The URL to fetch
            extract_text: Whether to extract plain text from HTML
            headers: Additional request headers
            timeout: Request timeout

        Returns:
            ToolResult with fetched content
        """
        if not AIOHTTP_AVAILABLE:
            return ToolResult.fail(
                "aiohttp is required for web fetching. "
                "Install with: pip install aiohttp"
            )

        # Check domain
        allowed, reason = self._config.is_domain_allowed(url)
        if not allowed:
            return ToolResult.fail(reason)

        # Prepare headers
        request_headers = {
            "User-Agent": self._config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        if headers:
            request_headers.update(headers)

        # Prepare SSL context
        ssl_context = None
        if not self._config.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        effective_timeout = timeout or self._config.timeout

        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout_obj = aiohttp.ClientTimeout(total=effective_timeout)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout_obj
            ) as session:
                async with session.get(
                    url,
                    headers=request_headers,
                    allow_redirects=self._config.follow_redirects,
                    max_redirects=self._config.max_redirects
                ) as response:
                    # Check content length
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self._config.max_content_size:
                        return ToolResult.fail(
                            f"Content too large: {content_length} bytes "
                            f"(max: {self._config.max_content_size})"
                        )

                    # Read content
                    content = await response.read()

                    if len(content) > self._config.max_content_size:
                        content = content[:self._config.max_content_size]

                    # Get content type
                    content_type = response.headers.get('Content-Type', '')

                    # Try to decode as text
                    text_content = None
                    encoding = response.charset or 'utf-8'
                    try:
                        text_content = content.decode(encoding, errors='replace')
                    except Exception:
                        pass

                    # Extract text if requested and content is HTML
                    extracted_text = None
                    if extract_text and text_content and 'html' in content_type.lower():
                        parser = SimpleHTMLTextExtractor()
                        parser.feed(text_content)
                        extracted_text = parser.get_text()

                    return ToolResult.ok(
                        extracted_text if extract_text else text_content,
                        url=str(response.url),
                        status_code=response.status,
                        content_type=content_type,
                        content_length=len(content),
                        encoding=encoding,
                        raw_html=text_content if extract_text else None
                    )

        except aiohttp.ClientError as e:
            return ToolResult.fail(f"HTTP error: {e}")
        except asyncio.TimeoutError:
            return ToolResult.fail(f"Request timed out after {effective_timeout}s")
        except Exception as e:
            return ToolResult.fail(f"Fetch error: {e}")


@dataclass
class SearchConfig:
    """Configuration for web search.

    Attributes:
        api_key: API key for the search service
        search_engine_id: Custom search engine ID (for Google)
        base_url: Base URL for the search API
        max_results: Default maximum results
    """
    api_key: Optional[str] = None
    search_engine_id: Optional[str] = None
    base_url: str = "https://www.googleapis.com/customsearch/v1"
    max_results: int = 10


class WebSearchTool(Tool):
    """Tool for performing web searches via API.

    This tool supports Google Custom Search API by default but
    can be configured for other search APIs.
    """

    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        fetch_config: Optional[FetchConfig] = None
    ):
        """Initialize the web search tool.

        Args:
            config: Search configuration
            fetch_config: Configuration for fetching search results
        """
        self._config = config or SearchConfig()
        self._fetch_config = fetch_config or FetchConfig()

        # Try to get API key from environment if not provided
        if not self._config.api_key:
            self._config.api_key = os.environ.get('GOOGLE_API_KEY')
        if not self._config.search_engine_id:
            self._config.search_engine_id = os.environ.get('GOOGLE_CSE_ID')

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Search the web using a search API.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                },
                "site": {
                    "type": "string",
                    "description": "Limit search to specific site"
                },
                "file_type": {
                    "type": "string",
                    "description": "Search for specific file type (e.g., 'pdf')"
                },
                "date_restrict": {
                    "type": "string",
                    "description": "Restrict by date (e.g., 'd7' for past week)"
                }
            },
            required=["query"],
            permissions=[ToolPermission.NETWORK]
        )

    async def execute(
        self,
        query: str,
        num_results: int = 10,
        site: Optional[str] = None,
        file_type: Optional[str] = None,
        date_restrict: Optional[str] = None
    ) -> ToolResult:
        """Perform a web search.

        Args:
            query: Search query
            num_results: Number of results
            site: Limit to specific site
            file_type: Search for file type
            date_restrict: Date restriction

        Returns:
            ToolResult with search results
        """
        if not AIOHTTP_AVAILABLE:
            return ToolResult.fail(
                "aiohttp is required for web search. "
                "Install with: pip install aiohttp"
            )

        if not self._config.api_key:
            return ToolResult.fail(
                "Search API key not configured. "
                "Set GOOGLE_API_KEY environment variable or provide in config."
            )

        if not self._config.search_engine_id:
            return ToolResult.fail(
                "Search engine ID not configured. "
                "Set GOOGLE_CSE_ID environment variable or provide in config."
            )

        # Build query
        search_query = query
        if site:
            search_query = f"site:{site} {search_query}"
        if file_type:
            search_query = f"filetype:{file_type} {search_query}"

        # Build API URL
        params = {
            "key": self._config.api_key,
            "cx": self._config.search_engine_id,
            "q": search_query,
            "num": min(num_results, self._config.max_results)
        }
        if date_restrict:
            params["dateRestrict"] = date_restrict

        query_string = "&".join(
            f"{k}={quote_plus(str(v))}" for k, v in params.items()
        )
        url = f"{self._config.base_url}?{query_string}"

        try:
            timeout = aiohttp.ClientTimeout(total=30.0)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        return ToolResult.fail(
                            f"Search API error ({response.status}): {error_text}"
                        )

                    data = await response.json()

            # Parse results
            results = []
            for item in data.get("items", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("link"),
                    "snippet": item.get("snippet"),
                    "display_url": item.get("displayLink")
                })

            return ToolResult.ok(
                results,
                query=query,
                total_results=data.get("searchInformation", {}).get("totalResults"),
                search_time=data.get("searchInformation", {}).get("searchTime")
            )

        except aiohttp.ClientError as e:
            return ToolResult.fail(f"HTTP error: {e}")
        except asyncio.TimeoutError:
            return ToolResult.fail("Search request timed out")
        except Exception as e:
            return ToolResult.fail(f"Search error: {e}")


class URLParserTool(Tool):
    """Tool for parsing and manipulating URLs."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="parse_url",
            description="Parse a URL into its components.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to parse"
                },
                "join_with": {
                    "type": "string",
                    "description": "Base URL to join with (for relative URLs)"
                }
            },
            required=["url"],
            permissions=[]
        )

    async def execute(
        self,
        url: str,
        join_with: Optional[str] = None
    ) -> ToolResult:
        """Parse a URL.

        Args:
            url: URL to parse
            join_with: Base URL for relative URLs

        Returns:
            ToolResult with URL components
        """
        try:
            # Join with base if provided
            if join_with:
                url = urljoin(join_with, url)

            parsed = urlparse(url)

            return ToolResult.ok({
                "url": url,
                "scheme": parsed.scheme,
                "netloc": parsed.netloc,
                "path": parsed.path,
                "params": parsed.params,
                "query": parsed.query,
                "fragment": parsed.fragment,
                "hostname": parsed.hostname,
                "port": parsed.port
            })

        except Exception as e:
            return ToolResult.fail(f"URL parse error: {e}")


# Convenience function to get web tools
def get_web_tools(
    fetch_config: Optional[FetchConfig] = None,
    search_config: Optional[SearchConfig] = None
) -> List[Tool]:
    """Get all web tools.

    Args:
        fetch_config: Configuration for web fetching
        search_config: Configuration for web search

    Returns:
        List of web tool instances
    """
    return [
        WebFetchTool(config=fetch_config),
        WebSearchTool(config=search_config, fetch_config=fetch_config),
        URLParserTool()
    ]
