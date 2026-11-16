"""
Unified Diff Parser for Pull Request changes.
Parses git diff output or GitHub patch text into structured file change records.
"""
from typing import List, Optional
from ingestion.git_loader import FileDiffItem


class PRDiffParser:
    """Parses standard unified diff syntax into structured FileDiffItem objects."""

    @staticmethod
    def parse_unified_diff(diff_text: str) -> List[FileDiffItem]:
        """Parses a multi-file unified diff string into FileDiffItem instances."""
        if not diff_text or not diff_text.strip():
            return []

        items: List[FileDiffItem] = []
        lines = diff_text.splitlines()

        curr_file: Optional[str] = None
        curr_patch_lines: List[str] = []
        change_type = "MODIFIED"

        for line in lines:
            if line.startswith("diff --git"):
                if curr_file and curr_patch_lines:
                    ptext = "\n".join(curr_patch_lines)
                    ins = ptext.count("\n+") - ptext.count("\n+++")
                    dels = ptext.count("\n-") - ptext.count("\n---")
                    items.append(FileDiffItem(
                        filepath=curr_file,
                        change_type=change_type,
                        insertions=max(0, ins),
                        deletions=max(0, dels),
                        patch=ptext
                    ))
                parts = line.split(" b/")
                curr_file = parts[-1] if len(parts) > 1 else "unknown"
                curr_patch_lines = [line]
                change_type = "MODIFIED"
            elif line.startswith("new file mode"):
                change_type = "ADDED"
                curr_patch_lines.append(line)
            elif line.startswith("deleted file mode"):
                change_type = "DELETED"
                curr_patch_lines.append(line)
            elif line.startswith("similarity index"):
                change_type = "RENAMED"
                curr_patch_lines.append(line)
            else:
                curr_patch_lines.append(line)

        if curr_file and curr_patch_lines:
            ptext = "\n".join(curr_patch_lines)
            ins = ptext.count("\n+") - ptext.count("\n+++")
            dels = ptext.count("\n-") - ptext.count("\n---")
            items.append(FileDiffItem(
                filepath=curr_file,
                change_type=change_type,
                insertions=max(0, ins),
                deletions=max(0, dels),
                patch=ptext
            ))

        return items
