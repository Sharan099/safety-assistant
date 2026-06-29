import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

import httpx
from loguru import logger

from crawler.allowlist import assert_url_allowed
from registry.storage_paths import staging_dir


class BaseCrawler(ABC):
    """Base interface for all regulation crawlers."""

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or str(staging_dir())

    @abstractmethod
    def crawl(self, mock: bool = True) -> List[Dict[str, Any]]:
        """
        Returns list of dicts:
          {"file_path": str, "metadata": dict, "source_url": str}
        """
        pass

    def _download_file(self, url: str, filename: str, authority: str) -> str:
        """Download to staging after allow-list check (HTTPS + verified cert)."""
        assert_url_allowed(url, authority)
        dest_path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Downloading {url} to {dest_path}")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AutoSafety-RAG/1.0; +https://github.com/local/registry)"
            )
        }
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=60.0, follow_redirects=True, verify=True) as client:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    Path(dest_path).write_bytes(response.content)
                logger.info(f"Downloaded successfully: {dest_path}")
                return dest_path
            except Exception as exc:
                last_err = exc
                wait = 2**attempt
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {exc}")
                time.sleep(wait)
        raise RuntimeError(f"Failed to download {url}: {last_err}") from last_err
