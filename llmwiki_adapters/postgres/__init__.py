"""Shared PostgreSQL adapters."""

from .wiki import WikiWriteResult, write_wiki_bundle_in_transaction

__all__ = ["WikiWriteResult", "write_wiki_bundle_in_transaction"]
