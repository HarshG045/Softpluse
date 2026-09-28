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


@dataclass
class FileDiffItem:
    """Represents a single file diff within a commit or working tree change."""
    filepath: str
    change_type: str  # "ADDED", "MODIFIED", "DELETED"
    insertions: int
    deletions: int
    patch: str


@dataclass
class CommitDetail:
    """Detailed information about a commit including parent hash and file diffs."""
    commit_hash: str
    short_hash: str
    author_name: str
    author_email: str
    timestamp: datetime
    message: str
    parent_hash: Optional[str]
    files_changed: List[FileDiffItem]
    total_insertions: int
    total_deletions: int
    is_bugfix: bool = False


@dataclass
class ContributorStats:
    """Detailed activity statistics for a repository contributor."""
    author_name: str
    author_email: str
    commit_count: int
    files_changed_count: int
    insertions: int
    deletions: int
    components_touched: List[str]
    last_activity: datetime
    first_activity: datetime
    commits: List[CommitRecord]


@dataclass
class BranchComparison:
    """Comparison metrics between two branches."""
    base_branch: str
    target_branch: str
    ahead_commits: int
    files_changed: List[str]
    insertions: int
    deletions: int
    affected_components: List[str]


class GitLoader:
    """Extracts granular commit history, churn stats, diffs, and contributor data from a local Git repository."""

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

    def get_commit_detail(self, commit_hash: str) -> Optional[CommitDetail]:
        """Extracts complete detail for a specific commit, including parent hash and file diffs."""
        if not self.is_git_repo():
            return None

        try:
            import git
            repo = git.Repo(str(self.repo_path))
            commit = repo.commit(commit_hash)

            parent_hash = commit.parents[0].hexsha if commit.parents else None
            diff_items: List[FileDiffItem] = []
            total_ins = 0
            total_dels = 0

            if commit.parents:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                for d in diffs:
                    fpath = d.b_path or d.a_path or "unknown"
                    ctype = "MODIFIED"
                    if d.new_file:
                        ctype = "ADDED"
                    elif d.deleted_file:
                        ctype = "DELETED"
                    elif d.renamed_file:
                        ctype = "RENAMED"

                    patch_text = ""
                    if d.diff:
                        try:
                            patch_text = d.diff.decode("utf-8", errors="replace") if isinstance(d.diff, bytes) else str(d.diff)
                        except Exception:
                            patch_text = ""

                    ins = patch_text.count("\n+") if patch_text else 0
                    dels = patch_text.count("\n-") if patch_text else 0
                    total_ins += ins
                    total_dels += dels

                    diff_items.append(FileDiffItem(
                        filepath=fpath,
                        change_type=ctype,
                        insertions=ins,
                        deletions=dels,
                        patch=patch_text
                    ))
            else:
                # Initial root commit: diff against empty tree
                stats = commit.stats
                for fpath, data in stats.files.items():
                    ins = data.get("insertions", 0)
                    dels = data.get("deletions", 0)
                    total_ins += ins
                    total_dels += dels
                    diff_items.append(FileDiffItem(
                        filepath=fpath,
                        change_type="ADDED",
                        insertions=ins,
                        deletions=dels,
                        patch=f"+ [Initial commit creation with {ins} lines]"
                    ))

            msg = commit.message.strip() if commit.message else ""
            dt = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)
            is_bugfix = any(kw in msg.lower() for kw in self.bug_keywords)

            return CommitDetail(
                commit_hash=commit.hexsha,
                short_hash=commit.hexsha[:7],
                author_name=commit.author.name or "Unknown",
                author_email=commit.author.email or "Unknown",
                timestamp=dt,
                message=msg,
                parent_hash=parent_hash,
                files_changed=diff_items,
                total_insertions=total_ins,
                total_deletions=total_dels,
                is_bugfix=is_bugfix
            )
        except Exception as e:
            logger.warning(f"Error extracting commit detail for {commit_hash}: {e}")
            return self._get_commit_detail_subprocess(commit_hash)

    def _get_commit_detail_subprocess(self, commit_hash: str) -> Optional[CommitDetail]:
        """Subprocess fallback for commit detail."""
        try:
            cmd = ["git", "show", "--format=%H|||%an|||%ae|||%ct|||%P|||%s", "--patch", commit_hash]
            res = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, check=True)
            output = res.stdout

            lines = output.split("\n")
            if not lines:
                return None

            header = lines[0].split("|||")
            chash = header[0]
            author = header[1] if len(header) > 1 else "Unknown"
            email = header[2] if len(header) > 2 else "Unknown"
            ts = int(header[3]) if len(header) > 3 and header[3].isdigit() else 0
            parents = header[4].split() if len(header) > 4 and header[4] else []
            parent_hash = parents[0] if parents else None
            msg = header[5] if len(header) > 5 else ""

            patch_content = "\n".join(lines[1:])
            diff_items: List[FileDiffItem] = []
            
            # Simple diff parser
            curr_file = None
            curr_patch_lines: List[str] = []
            for pline in lines[1:]:
                if pline.startswith("diff --git"):
                    if curr_file and curr_patch_lines:
                        ptext = "\n".join(curr_patch_lines)
                        diff_items.append(FileDiffItem(
                            filepath=curr_file,
                            change_type="MODIFIED",
                            insertions=ptext.count("\n+"),
                            deletions=ptext.count("\n-"),
                            patch=ptext
                        ))
                    parts = pline.split(" b/")
                    curr_file = parts[-1] if len(parts) > 1 else "file"
                    curr_patch_lines = [pline]
                else:
                    curr_patch_lines.append(pline)

            if curr_file and curr_patch_lines:
                ptext = "\n".join(curr_patch_lines)
                diff_items.append(FileDiffItem(
                    filepath=curr_file,
                    change_type="MODIFIED",
                    insertions=ptext.count("\n+"),
                    deletions=ptext.count("\n-"),
                    patch=ptext
                ))

            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            total_ins = sum(d.insertions for d in diff_items)
            total_dels = sum(d.deletions for d in diff_items)

            return CommitDetail(
                commit_hash=chash,
                short_hash=chash[:7],
                author_name=author,
                author_email=email,
                timestamp=dt,
                message=msg,
                parent_hash=parent_hash,
                files_changed=diff_items,
                total_insertions=total_ins,
                total_deletions=total_dels,
                is_bugfix=any(kw in msg.lower() for kw in self.bug_keywords)
            )
        except Exception as e:
            logger.error(f"Subprocess failed to extract detail for {commit_hash}: {e}")
            return None

    def get_contributor_stats(self) -> List[ContributorStats]:
        """Calculates real contributor metrics across entire commit history."""
        commits = self.get_commit_history()
        if not commits:
            return []

        author_map: Dict[str, Dict] = {}
        for c in commits:
            key = c.author_name.strip()
            if key not in author_map:
                author_map[key] = {
                    "email": c.author_email,
                    "commits": [],
                    "files": set(),
                    "insertions": 0,
                    "deletions": 0,
                    "first_activity": c.timestamp,
                    "last_activity": c.timestamp
                }

            author_map[key]["commits"].append(c)
            author_map[key]["files"].update(c.files_changed)
            author_map[key]["insertions"] += c.insertions
            author_map[key]["deletions"] += c.deletions
            if c.timestamp > author_map[key]["last_activity"]:
                author_map[key]["last_activity"] = c.timestamp
            if c.timestamp < author_map[key]["first_activity"]:
                author_map[key]["first_activity"] = c.timestamp

        results: List[ContributorStats] = []
        for name, data in author_map.items():
            results.append(ContributorStats(
                author_name=name,
                author_email=data["email"],
                commit_count=len(data["commits"]),
                files_changed_count=len(data["files"]),
                insertions=data["insertions"],
                deletions=data["deletions"],
                components_touched=sorted(list(data["files"])),
                last_activity=data["last_activity"],
                first_activity=data["first_activity"],
                commits=data["commits"]
            ))

        # Sort by commit count descending
        results.sort(key=lambda x: x.commit_count, reverse=True)
        return results

    def get_component_git_history(self, filepath: str) -> List[CommitRecord]:
        """Returns all commit records that modified a specific file or component."""
        commits = self.get_commit_history()
        norm_path = filepath.replace("\\", "/").lower()
        matched: List[CommitRecord] = []

        for c in reversed(commits):  # Newest first
            for f in c.files_changed:
                f_norm = f.replace("\\", "/").lower()
                if norm_path.endswith(f_norm) or f_norm.endswith(norm_path) or norm_path == f_norm:
                    matched.append(c)
                    break

        return matched

    def get_branches(self) -> Tuple[str, List[str]]:
        """Returns (current_branch, list_of_all_branches)."""
        if not self.is_git_repo():
            return ("main", ["main"])

        try:
            import git
            repo = git.Repo(str(self.repo_path))
            active = repo.active_branch.name if not repo.head.is_detached else "detached-HEAD"
            branches = [h.name for h in repo.heads]
            if not branches:
                branches = [active]
            return (active, branches)
        except Exception:
            try:
                res = subprocess.run(["git", "branch", "--show-current"], cwd=str(self.repo_path), capture_output=True, text=True)
                active = res.stdout.strip() or "main"
                res_all = subprocess.run(["git", "branch"], cwd=str(self.repo_path), capture_output=True, text=True)
                branches = [line.strip().replace("* ", "") for line in res_all.stdout.split("\n") if line.strip()]
                return (active, branches if branches else [active])
            except Exception:
                return ("main", ["main"])

    def compare_branches(self, base_branch: str, target_branch: str) -> BranchComparison:
        """Compares two branches in terms of commits ahead, files changed, and churn."""
        if not self.is_git_repo() or base_branch == target_branch:
            return BranchComparison(
                base_branch=base_branch,
                target_branch=target_branch,
                ahead_commits=0,
                files_changed=[],
                insertions=0,
                deletions=0,
                affected_components=[]
            )

        try:
            cmd = ["git", "diff", "--numstat", f"{base_branch}..{target_branch}"]
            res = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True)
            files = []
            ins_tot = 0
            del_tot = 0
            for line in res.stdout.split("\n"):
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    ins_tot += int(parts[0]) if parts[0].isdigit() else 0
                    del_tot += int(parts[1]) if parts[1].isdigit() else 0
                    files.append(parts[2])

            cmd_count = ["git", "rev-list", "--count", f"{base_branch}..{target_branch}"]
            res_c = subprocess.run(cmd_count, cwd=str(self.repo_path), capture_output=True, text=True)
            ahead = int(res_c.stdout.strip()) if res_c.stdout.strip().isdigit() else 0

            return BranchComparison(
                base_branch=base_branch,
                target_branch=target_branch,
                ahead_commits=ahead,
                files_changed=files,
                insertions=ins_tot,
                deletions=del_tot,
                affected_components=files
            )
        except Exception as e:
            logger.warning(f"Branch comparison failed: {e}")
            return BranchComparison(
                base_branch=base_branch,
                target_branch=target_branch,
                ahead_commits=0,
                files_changed=[],
                insertions=0,
                deletions=0,
                affected_components=[]
            )

    def get_working_tree_diff(self) -> List[FileDiffItem]:
        """Returns diffs for uncommitted working tree changes."""
        if not self.is_git_repo():
            return []

        try:
            import git
            repo = git.Repo(str(self.repo_path))
            diffs = repo.head.commit.diff(None, create_patch=True)
            items: List[FileDiffItem] = []
            for d in diffs:
                fpath = d.b_path or d.a_path or "unknown"
                ctype = "MODIFIED"
                if d.new_file:
                    ctype = "ADDED"
                elif d.deleted_file:
                    ctype = "DELETED"
                ptext = d.diff.decode("utf-8", errors="replace") if isinstance(d.diff, bytes) else str(d.diff or "")
                items.append(FileDiffItem(
                    filepath=fpath,
                    change_type=ctype,
                    insertions=ptext.count("\n+"),
                    deletions=ptext.count("\n-"),
                    patch=ptext
                ))
            return items
        except Exception:
            return []

    def safe_commit(self, message: str, author_name: str = "Developer", author_email: str = "dev@softwarepulse.ai") -> Tuple[bool, str]:
        """
        Creates a Git commit with user confirmation.
        Returns (success: bool, status_or_hash: str).
        """
        if not self.is_git_repo():
            return (False, "Target directory is not a valid Git repository.")

        if not message.strip():
            return (False, "Commit message cannot be empty.")

        try:
            # Stage all current working changes
            subprocess.run(["git", "add", "-A"], cwd=str(self.repo_path), check=True, capture_output=True)
            
            # Commit with specified author
            cmd = [
                "git",
                "-c", f"user.name={author_name}",
                "-c", f"user.email={author_email}",
                "commit",
                "-m", message.strip()
            ]
            res = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True)
            if res.returncode != 0:
                if "nothing to commit" in res.stdout or "nothing to commit" in res.stderr:
                    return (False, "No changes detected in working tree to commit.")
                return (False, f"Git commit failed: {res.stderr.strip() or res.stdout.strip()}")

            # Get new commit hash
            rev_res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(self.repo_path), capture_output=True, text=True)
            new_hash = rev_res.stdout.strip()
            return (True, new_hash)
        except Exception as e:
            return (False, f"Commit error: {str(e)}")

    def safe_push(self, remote_name: str = "origin", branch_name: Optional[str] = None) -> Tuple[bool, str]:
        """
        Attempts to push to remote repository.
        Returns (success: bool, message: str).
        """
        if not self.is_git_repo():
            return (False, "Target is not a Git repository.")

        curr_branch, _ = self.get_branches()
        target = branch_name or curr_branch

        try:
            # Check if remote exists
            remotes = subprocess.run(["git", "remote"], cwd=str(self.repo_path), capture_output=True, text=True)
            if remote_name not in remotes.stdout.split():
                return (False, f"Push unavailable — remote '{remote_name}' is not configured.")

            cmd = ["git", "push", remote_name, target]
            res = subprocess.run(cmd, cwd=str(self.repo_path), capture_output=True, text=True, timeout=15)
            if res.returncode == 0:
                return (True, f"Successfully pushed to {remote_name}/{target}")
            else:
                return (False, f"Push unavailable — configure repository authentication or credentials. ({res.stderr.strip()})")
        except subprocess.TimeoutExpired:
            return (False, "Push timed out. Remote authentication might be required.")
        except Exception as e:
            return (False, f"Push failed: {str(e)}")

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
