"""
Data classes and helper functions for repository metadata representation.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class RepositoryMetadata:
    """Stores high-level repository metadata extracted during ingestion."""
    name: str
    path: str
    default_branch: str
    commit_count: int
    contributor_count: int
    file_count: int
    languages: Dict[str, int]  # Extension -> count
    first_commit_date: Optional[datetime] = None
    latest_commit_date: Optional[datetime] = None
    is_remote: bool = False
    remote_url: Optional[str] = None
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Serialize metadata to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "default_branch": self.default_branch,
            "commit_count": self.commit_count,
            "contributor_count": self.contributor_count,
            "file_count": self.file_count,
            "languages": self.languages,
            "first_commit_date": self.first_commit_date.isoformat() if self.first_commit_date else None,
            "latest_commit_date": self.latest_commit_date.isoformat() if self.latest_commit_date else None,
            "is_remote": self.is_remote,
            "remote_url": self.remote_url,
            "analyzed_at": self.analyzed_at.isoformat()
        }
