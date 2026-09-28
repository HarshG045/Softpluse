"""
SoftwarePulse Ingestion Package.
Handles local repository loading, remote cloning, git history extraction, and metadata extraction.
"""
from .repository_loader import RepositoryLoader
from .git_loader import GitLoader
from .repository_metadata import RepositoryMetadata

__all__ = ["RepositoryLoader", "GitLoader", "RepositoryMetadata"]
