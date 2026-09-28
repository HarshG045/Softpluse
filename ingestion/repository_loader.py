"""
RepositoryLoader: Handles repository ingestion, remote git cloning, validation, and metadata extraction.
"""
import logging
import os
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import RAW_DATA_DIR, SUPPORTED_PYTHON_EXTENSIONS
from .git_loader import GitLoader
from .repository_metadata import RepositoryMetadata

logger = logging.getLogger(__name__)


class RepositoryLoader:
    """Manages repository loading from local directory or remote GitHub URL."""

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR

    @staticmethod
    def is_github_url(url_or_path: str) -> bool:
        """Determines if a given string is a Git/GitHub URL."""
        github_pattern = r"^(https?:\/\/|git@)(github\.com|gitlab\.com|bitbucket\.org)[\/:].+\.git$"
        return bool(re.match(github_pattern, url_or_path.strip())) or "github.com/" in url_or_path

    def clone_remote_repo(self, repo_url: str) -> Path:
        """Clones a remote git repository into a safe raw data subfolder."""
        clean_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
        clean_name = re.sub(r"[^\w\-]", "_", clean_name)
        target_dir = self.raw_data_dir / clean_name

        if target_dir.exists() and (target_dir / ".git").exists():
            logger.info(f"Using already cloned repository at {target_dir}")
            return target_dir

        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        logger.info(f"Cloning {repo_url} into {target_dir}...")
        try:
            import git
            git.Repo.clone_from(repo_url, str(target_dir), depth=500)
            return target_dir
        except Exception as e:
            logger.warning(f"GitPython clone failed: {e}. Attempting subprocess git clone...")
            res = subprocess.run(["git", "clone", "--depth", "500", repo_url, str(target_dir)],
                                 capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Failed to clone repository from {repo_url}: {res.stderr}")
            return target_dir

    def load_repository(self, path_or_url: str) -> Tuple[Path, RepositoryMetadata]:
        """
        Loads a repository from either a local filesystem path or remote URL.
        Returns the resolved local repository Path and extracted RepositoryMetadata.
        """
        is_remote = self.is_github_url(path_or_url)
        if is_remote:
            repo_path = self.clone_remote_repo(path_or_url)
            remote_url = path_or_url
        else:
            repo_path = Path(path_or_url).resolve()
            remote_url = None

        if not repo_path.exists() or not repo_path.is_dir():
            raise FileNotFoundError(f"Repository directory does not exist: {repo_path}")

        metadata = self.extract_metadata(repo_path, is_remote=is_remote, remote_url=remote_url)
        return repo_path, metadata

    def extract_metadata(self, repo_path: Path, is_remote: bool = False, remote_url: Optional[str] = None) -> RepositoryMetadata:
        """Scans the repository to extract high-level summary statistics."""
        name = repo_path.name
        git_loader = GitLoader(str(repo_path))
        is_git = git_loader.is_git_repo()

        # Extract file distribution and languages
        languages: Dict[str, int] = Counter()
        file_count = 0

        # Exclude hidden folders (.git, .venv, etc.)
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", "__pycache__", "build", "dist")]
            for f in files:
                ext = Path(f).suffix.lower() or "other"
                languages[ext] += 1
                file_count += 1

        # Git statistics
        default_branch = "unknown"
        commits = []
        contributors = set()
        first_commit_date = None
        latest_commit_date = None

        if is_git:
            try:
                import git
                r = git.Repo(str(repo_path))
                try:
                    default_branch = r.active_branch.name
                except Exception:
                    default_branch = "main" if "main" in [h.name for h in r.heads] else ("master" if "master" in [h.name for h in r.heads] else "HEAD")
            except Exception:
                default_branch = "main"

            commits = git_loader.get_commit_history()
            for c in commits:
                contributors.add(c.author_email or c.author_name)
            
            if commits:
                first_commit_date = commits[0].timestamp
                latest_commit_date = commits[-1].timestamp

        return RepositoryMetadata(
            name=name,
            path=str(repo_path),
            default_branch=default_branch,
            commit_count=len(commits),
            contributor_count=max(len(contributors), 1 if commits else 0),
            file_count=file_count,
            languages=dict(languages.most_common(10)),
            first_commit_date=first_commit_date,
            latest_commit_date=latest_commit_date,
            is_remote=is_remote,
            remote_url=remote_url,
            analyzed_at=datetime.now(timezone.utc)
        )
