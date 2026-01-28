"""
Framework Source Fetcher.

This module provides tools for fetching framework source code from
various sources including GitHub, URLs, and PyPI.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

# Optional imports
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class FrameworkFetcher:
    """
    Fetch framework source code from various sources.

    Supports fetching from:
    - GitHub repositories (via git clone or API)
    - Direct URLs
    - PyPI packages

    Attributes:
        cache_dir: Directory for caching downloaded sources
        github_token: Optional GitHub token for API access

    Example:
        ```python
        fetcher = FrameworkFetcher()

        # Fetch from GitHub
        path = await fetcher.fetch_github("https://github.com/langchain-ai/langchain")

        # Fetch from PyPI
        path = await fetcher.fetch_pypi("langchain")

        # List Python files
        files = await fetcher.list_files(path, [".py"])
        ```
    """

    # Default extensions to look for
    DEFAULT_EXTENSIONS = [".py", ".pyi"]

    # Directories to skip when listing files
    SKIP_DIRS = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".eggs",
        "build",
        "dist",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
    ):
        """
        Initialize the framework fetcher.

        Args:
            cache_dir: Directory for caching downloads
            github_token: GitHub token for API access
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "agent_frameworks_cache"
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_github(
        self,
        repo_url: str,
        branch: str = "main",
        use_cache: bool = True,
    ) -> Path:
        """
        Fetch a GitHub repository.

        Clones the repository to the cache directory and returns
        the path to the cloned code.

        Args:
            repo_url: GitHub repository URL
            branch: Branch to checkout
            use_cache: Whether to use cached clone

        Returns:
            Path to the cloned repository
        """
        # Parse repo URL
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")

        owner = path_parts[0]
        repo = path_parts[1].replace(".git", "")

        # Create cache path
        repo_cache_dir = self.cache_dir / "github" / owner / repo

        # Check cache
        if use_cache and repo_cache_dir.exists():
            # Pull latest changes
            await self._git_pull(repo_cache_dir, branch)
            return repo_cache_dir

        # Clone repository
        repo_cache_dir.parent.mkdir(parents=True, exist_ok=True)

        clone_url = f"https://github.com/{owner}/{repo}.git"

        await self._git_clone(clone_url, repo_cache_dir, branch)

        return repo_cache_dir

    async def fetch_url(
        self,
        url: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Fetch content from a URL.

        Downloads the content and returns it as a string.

        Args:
            url: URL to fetch from
            filename: Optional filename for caching

        Returns:
            Content as string
        """
        if HTTPX_AVAILABLE:
            return await self._fetch_with_httpx(url)
        elif AIOHTTP_AVAILABLE:
            return await self._fetch_with_aiohttp(url)
        else:
            return await self._fetch_with_urllib(url)

    async def fetch_pypi(
        self,
        package_name: str,
        version: Optional[str] = None,
    ) -> Path:
        """
        Fetch a package from PyPI.

        Downloads and extracts the package source code.

        Args:
            package_name: Name of the PyPI package
            version: Specific version to fetch (latest if None)

        Returns:
            Path to the extracted package
        """
        # Create cache path
        version_str = version or "latest"
        package_cache_dir = self.cache_dir / "pypi" / package_name / version_str

        if package_cache_dir.exists():
            return package_cache_dir

        package_cache_dir.mkdir(parents=True, exist_ok=True)

        # Use pip to download the package
        pip_args = [
            "pip",
            "download",
            "--no-deps",
            "--no-binary", ":all:",
            "-d", str(package_cache_dir),
        ]

        if version:
            pip_args.append(f"{package_name}=={version}")
        else:
            pip_args.append(package_name)

        process = await asyncio.create_subprocess_exec(
            *pip_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"Failed to download {package_name}: {stderr.decode()}"
            )

        # Find and extract the downloaded archive
        for file in package_cache_dir.iterdir():
            if file.suffix == ".gz" or file.suffix == ".zip":
                await self._extract_archive(file, package_cache_dir / "src")
                return package_cache_dir / "src"

        return package_cache_dir

    async def list_files(
        self,
        path: Path,
        extensions: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        List files in a directory with given extensions.

        Recursively lists all files matching the specified extensions,
        excluding common non-source directories.

        Args:
            path: Directory to search
            extensions: File extensions to include

        Returns:
            List of matching file paths
        """
        extensions = extensions or self.DEFAULT_EXTENSIONS
        files: List[Path] = []

        if not path.exists():
            return files

        if path.is_file():
            if path.suffix in extensions:
                files.append(path)
            return files

        for item in path.rglob("*"):
            # Skip unwanted directories
            if any(skip in item.parts for skip in self.SKIP_DIRS):
                continue

            if item.is_file() and item.suffix in extensions:
                files.append(item)

        return sorted(files)

    async def read_file(
        self,
        path: Path,
        encoding: str = "utf-8",
    ) -> str:
        """
        Read a file's content.

        Args:
            path: Path to the file
            encoding: File encoding

        Returns:
            File content as string
        """
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Try different encodings
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    return path.read_text(encoding=enc)
                except UnicodeDecodeError:
                    continue
            raise

    async def get_repo_info(
        self,
        repo_url: str,
    ) -> Dict[str, Any]:
        """
        Get information about a GitHub repository.

        Uses the GitHub API to fetch repository metadata.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Repository information dictionary
        """
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")

        owner = path_parts[0]
        repo = path_parts[1].replace(".git", "")

        api_url = f"https://api.github.com/repos/{owner}/{repo}"

        headers = {"Accept": "application/vnd.github.v3+json"}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        content = await self.fetch_url(api_url)

        import json
        return json.loads(content)

    async def _git_clone(
        self,
        url: str,
        target: Path,
        branch: str,
    ) -> None:
        """Clone a git repository."""
        args = [
            "git",
            "clone",
            "--depth", "1",
            "--branch", branch,
            url,
            str(target),
        ]

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            # Try without branch specification (default branch)
            args = [
                "git",
                "clone",
                "--depth", "1",
                url,
                str(target),
            ]

            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Failed to clone {url}: {stderr.decode()}")

    async def _git_pull(
        self,
        repo_path: Path,
        branch: str,
    ) -> None:
        """Pull latest changes from a git repository."""
        # Checkout branch
        checkout_process = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "checkout", branch,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await checkout_process.communicate()

        # Pull changes
        pull_process = await asyncio.create_subprocess_exec(
            "git", "-C", str(repo_path), "pull", "--ff-only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await pull_process.communicate()

    async def _fetch_with_httpx(self, url: str) -> str:
        """Fetch URL using httpx."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def _fetch_with_aiohttp(self, url: str) -> str:
        """Fetch URL using aiohttp."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.text()

    async def _fetch_with_urllib(self, url: str) -> str:
        """Fetch URL using urllib (sync, wrapped in executor)."""
        import urllib.request

        loop = asyncio.get_event_loop()

        def fetch():
            with urllib.request.urlopen(url) as response:
                return response.read().decode("utf-8")

        return await loop.run_in_executor(None, fetch)

    async def _extract_archive(
        self,
        archive_path: Path,
        target_dir: Path,
    ) -> None:
        """Extract an archive file."""
        target_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == ".gz" and ".tar" in archive_path.name:
            # tar.gz file
            import tarfile

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: tarfile.open(archive_path, "r:gz").extractall(target_dir),
            )

        elif archive_path.suffix == ".zip":
            # zip file
            import zipfile

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: zipfile.ZipFile(archive_path, "r").extractall(target_dir),
            )

    def clear_cache(self) -> None:
        """Clear the cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_size(self) -> int:
        """Get total size of cached files in bytes."""
        total = 0
        for file in self.cache_dir.rglob("*"):
            if file.is_file():
                total += file.stat().st_size
        return total
