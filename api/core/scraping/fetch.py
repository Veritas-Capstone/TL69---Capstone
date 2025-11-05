"""DEPRECATED: `api.core.scraping` moved to `api.webscraping`.

This module is a small backwards-compatibility shim. Importing from
`api.core.scraping` will re-export the new implementation. The file is
kept deliberately tiny so it can be removed as soon as the rest of the
codebase is migrated.
"""

from api.webscraping.fetch import fetch, FetchResult, canonicalize_url  # re-export

__all__ = ["fetch", "FetchResult", "canonicalize_url"]
