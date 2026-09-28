"""
GitLoader: Extracts commit logs, file changes, insertions, deletions, author data, and timestamps.
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class CommitRecord:
    """Represents a single Git commit record."""
    commit_hash: str
    author_name: str
    author_email: str
    timestamp: datetime
    message: str
    files_changed: List[str]
    insertions: int
    deletions: int
    is_bugfix: bool = False


class GitLoader:
    """Extracts granular commit history and churn stats from a local Git repository."""

    def __init__(self, repo_path: str, bug_keywords: Optional[List[str]] = None):
        self.repo_path = Path(repo_path).resolve()
        self.bug_keywords = [kw.lower() for kw in (bug_keywords or [
            "fix", "bug", "bugfix", "defect", "patch", "error", 
            "crash", "failure", "issue", "resolve", "close", "regression"
        ])]

    def is_git_repo(self) -> bool:
        """Checks if the path is a valid Git repository."""
        return (self.repo_path / ".git").exists()

    def get_commit_history(self, max_commits: Optional[int] = None) -> List[CommitRecord]:
        """
        Extracts commit history using GitPython if available, otherwise Git subprocess.
        Returns a list of CommitRecord ordered from oldest to newest.
        """
        if not self.is_git_repo():
            logger.warning(f"Path {self.repo_path} is not a git repository.")
            return []

        try:
            import git
            repo = git.Repo(str(self.repo_path))
            commits = list(repo.iter_commits())
            commits.reverse()  # chronological: oldest first

            if max_commits and len(commits) > max_commits:
                commits = commits[-max_commits:]

            records: List[CommitRecord] = []
            for commit in commits:
                files_changed = []
                insertions = 0
                deletions = 0

                try:
                    stats = commit.stats
                    files_changed = list(stats.files.keys())
                    insertions = stats.total.get("insertions", 0)
                    deletions = stats.total.get("deletions", 0)
                except Exception as e:
                    logger.debug(f"Could not extract commit stats for {commit.hexsha}: {e}")

                msg = commit.message.strip() if commit.message else ""
                msg_lower = msg.lower()
                is_bugfix = any(kw in msg_lower for kw in self.bug_keywords)

                dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)

                record = CommitRecord(
                    commit_hash=commit.hexsha,
                    author_name=commit.author.name or "Unknown",
                    author_email=commit.author.email or "Unknown",
                    timestamp=dt,
                    message=msg,
                    files_changed=files_changed,
                    insertions=insertions,
                    deletions=deletions,
                    is_bugfix=is_bugfix
                )
                records.append(record)

            return records

        except Exception as e:
            logger.warning(f"GitPython extraction failed ({e}), falling back to git CLI subprocess.")
            return self._extract_via_subprocess(max_commits)

    def _extract_via_subprocess(self, max_commits: Optional[int] = None) -> List[CommitRecord]:
        """Fallback method using standard Git CLI commands."""
        try:
            cmd = ["git", "log", "--reverse", "--pretty=format:%H|||%an|||%ae|||%ct|||%s", "--numstat"]
            res = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=True)
            output = res.stdout.strip()

            if not output:
                return []

            records: List[CommitRecord] = []
            current_commit: Optional[CommitRecord] = None

            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue

                if "|||" in line:
                    if current_commit:
                        records.append(current_commit)
                    parts = line.split("|||")
                    chash = parts[0]
                    author = parts[1] if len(parts) > 1 else "Unknown"
                    email = parts[2] if len(parts) > 2 else "Unknown"
                    ts = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
                    msg = parts[4] if len(parts) > 4 else ""
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    msg_lower = msg.lower()
                    is_bugfix = any(kw in msg_lower for kw in self.bug_keywords)

                    current_commit = CommitRecord(
                        commit_hash=chash,
                        author_name=author,
                        author_email=email,
                        timestamp=dt,
                        message=msg,
                        files_changed=[],
                        insertions=0,
                        deletions=0,
                        is_bugfix=is_bugfix
                    )
                else:
                    # numstat line: <ins>\t<del>\t<filepath>
                    numstat_parts = line.split("\t")
                    if len(numstat_parts) >= 3 and current_commit:
                        ins = int(numstat_parts[0]) if numstat_parts[0].isdigit() else 0
                        dels = int(numstat_parts[1]) if numstat_parts[1].isdigit() else 0
                        fpath = numstat_parts[2]
                        current_commit.insertions += ins
                        current_commit.deletions += dels
                        current_commit.files_changed.append(fpath)

            if current_commit:
                records.append(current_commit)

            if max_commits and len(records) > max_commits:
                records = records[-max_commits:]

            return records
        except Exception as err:
            logger.error(f"Git subprocess extraction failed: {err}")
            return []
