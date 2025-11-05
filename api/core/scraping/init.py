"""DEPRECATED package: `api.core.scraping` moved to `api.webscraping`.

This package now re-exports the new implementation. Remove this package
once all imports have been migrated to `api.webscraping`.
"""

from api.webscraping.fetch import fetch, FetchResult, canonicalize_url

__all__ = ["fetch", "FetchResult", "canonicalize_url"]
