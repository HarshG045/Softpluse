"""
GitHubPRClient: Safe, read-only GitHub API client for fetching pull request diffs,
file changes, and commit metadata.
"""
import json
import logging
import os
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GitHubPRMetadata:
    """Metadata extracted for a GitHub Pull Request."""
    owner: str
    repo: str
    pr_number: int
    title: str
    author: str
    state: str
    base_ref: str
    head_ref: str
    base_sha: str
    head_sha: str
    additions: int
    deletions: int
    changed_files_count: int
    raw_diff: str = ""
    files_changed: List[str] = field(default_factory=list)


class GitHubPRClient:
    """Interacts with the GitHub REST API to fetch pull request metadata and unified diffs."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_TOKEN")

    @staticmethod
    def parse_pr_url(url: str) -> Optional[Tuple[str, str, int]]:
        """Parses a GitHub pull request URL into (owner, repo, pr_number)."""
        pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1), match.group(2), int(match.group(3))
        return None

    def fetch_pr(self, url_or_tuple) -> Optional[GitHubPRMetadata]:
        """Fetches PR details using GitHub REST API."""
        if isinstance(url_or_tuple, str):
            parsed = self.parse_pr_url(url_or_tuple)
            if not parsed:
                logger.error(f"Invalid GitHub PR URL: {url_or_tuple}")
                return None
            owner, repo, pr_num = parsed
        else:
            owner, repo, pr_num = url_or_tuple

        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SoftwarePulse-RiskAnalyzer/1.0"
        }
        if self.token:
            headers["Authorization"] = f"token {self.token}"

        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}"

        try:
            req = urllib.request.Request(api_url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Also fetch diff patch
            diff_headers = {**headers, "Accept": "application/vnd.github.v3.diff"}
            diff_req = urllib.request.Request(api_url, headers=diff_headers)
            raw_diff = ""
            try:
                with urllib.request.urlopen(diff_req, timeout=15) as d_resp:
                    raw_diff = d_resp.read().decode("utf-8", errors="replace")
            except Exception as e:
                logger.debug(f"Could not retrieve raw diff for PR #{pr_num}: {e}")

            # Also fetch files list
            files_url = f"{api_url}/files"
            files_req = urllib.request.Request(files_url, headers=headers)
            files_list = []
            try:
                with urllib.request.urlopen(files_req, timeout=15) as f_resp:
                    f_data = json.loads(f_resp.read().decode("utf-8"))
                    files_list = [item["filename"] for item in f_data]
            except Exception as e:
                logger.debug(f"Could not retrieve files list for PR #{pr_num}: {e}")

            return GitHubPRMetadata(
                owner=owner,
                repo=repo,
                pr_number=pr_num,
                title=data.get("title", f"PR #{pr_num}"),
                author=data.get("user", {}).get("login", "Unknown"),
                state=data.get("state", "open"),
                base_ref=data.get("base", {}).get("ref", "main"),
                head_ref=data.get("head", {}).get("ref", "feature"),
                base_sha=data.get("base", {}).get("sha", ""),
                head_sha=data.get("head", {}).get("sha", ""),
                additions=data.get("additions", 0),
                deletions=data.get("deletions", 0),
                changed_files_count=data.get("changed_files", len(files_list)),
                raw_diff=raw_diff,
                files_changed=files_list
            )

        except urllib.error.HTTPError as e:
            if e.code == 403:
                logger.warning(f"GitHub API rate limit exceeded or authorization required: {e}")
            elif e.code == 404:
                logger.warning(f"GitHub PR #{pr_num} not found on {owner}/{repo}: {e}")
            else:
                logger.warning(f"GitHub API error ({e.code}): {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching GitHub PR: {e}")
            return None
