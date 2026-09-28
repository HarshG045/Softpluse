"""
CodeAnalyzer: Extracts static metrics using Python AST (LOC, cyclomatic complexity, classes, functions, methods, imports).
"""
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Stores static code metrics for a single source file/component."""
    filepath: str
    loc: int = 0
    functions: int = 0
    classes: int = 0
    complexity: int = 1
    imports: int = 0
    methods: int = 0
    branches: int = 0
    comment_ratio: float = 0.0
    imported_modules: Set[str] = field(default_factory=set)
    has_parse_error: bool = False
    parse_error_msg: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "filepath": self.filepath,
            "loc": self.loc,
            "functions": self.functions,
            "classes": self.classes,
            "complexity": self.complexity,
            "imports": self.imports,
            "methods": self.methods,
            "branches": self.branches,
            "comment_ratio": round(self.comment_ratio, 3),
            "imported_modules": list(self.imported_modules),
            "has_parse_error": self.has_parse_error
        }


class CyclomaticComplexityVisitor(ast.NodeVisitor):
    """Calculates McCabe Cyclomatic Complexity from Python AST."""
    def __init__(self):
        self.complexity = 1
        self.branches = 0

    def visit_If(self, node):
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Each 'and' / 'or' adds an additional branch condition
        self.complexity += max(len(node.values) - 1, 1)
        self.branches += max(len(node.values) - 1, 1)
        self.generic_visit(node)

    def visit_IfExp(self, node):
        # Ternary conditional expressions (x if c else y)
        self.complexity += 1
        self.branches += 1
        self.generic_visit(node)

    def visit_Match(self, node):
        # Python 3.10+ pattern matching
        if hasattr(node, "cases"):
            self.complexity += len(node.cases)
            self.branches += len(node.cases)
        self.generic_visit(node)


class CodeAnalyzer:
    """Analyzes source code files within a repository using Python AST."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def analyze_file(self, rel_filepath: str) -> CodeMetrics:
        """Parses a single python file and extracts all code metrics."""
        full_path = self.repo_path / rel_filepath
        metrics = CodeMetrics(filepath=rel_filepath.replace("\\", "/"))

        if not full_path.exists() or not full_path.is_file():
            return metrics

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.splitlines()
            metrics.loc = len([line for line in lines if line.strip() and not line.strip().startswith("#")])
            
            comment_lines = len([line for line in lines if line.strip().startswith("#")])
            total_lines = len(lines)
            metrics.comment_ratio = (comment_lines / total_lines) if total_lines > 0 else 0.0

            if not content.strip():
                return metrics

            tree = ast.parse(content, filename=str(full_path))

            # Calculate cyclomatic complexity
            cc_visitor = CyclomaticComplexityVisitor()
            cc_visitor.visit(tree)
            metrics.complexity = cc_visitor.complexity
            metrics.branches = cc_visitor.branches

            # Extract classes, methods, functions, and imports
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    metrics.classes += 1
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            metrics.methods += 1

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    metrics.functions += 1

                elif isinstance(node, ast.Import):
                    metrics.imports += len(node.names)
                    for alias in node.names:
                        metrics.imported_modules.add(alias.name)
                        metrics.imported_modules.add(alias.name.split(".")[0])
                        metrics.imported_modules.add(alias.name.split(".")[-1])

                elif isinstance(node, ast.ImportFrom):
                    metrics.imports += 1
                    if node.module:
                        metrics.imported_modules.add(node.module)
                        metrics.imported_modules.add(node.module.split(".")[0])
                        metrics.imported_modules.add(node.module.split(".")[-1])


            # Top-level standalone functions vs total functions
            # In our metrics: functions = total function definitions, methods = class-bound functions

        except SyntaxError as e:
            metrics.has_parse_error = True
            metrics.parse_error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.debug(f"Syntax error in {rel_filepath}: {e}")
        except Exception as e:
            metrics.has_parse_error = True
            metrics.parse_error_msg = str(e)
            logger.debug(f"Failed to parse {rel_filepath}: {e}")

        return metrics

    def analyze_repository(self, filepaths: Optional[List[str]] = None) -> Dict[str, CodeMetrics]:
        """
        Analyzes all Python files in the repository.
        Returns a dict mapping normalized relative filepath to CodeMetrics.
        """
        results: Dict[str, CodeMetrics] = {}

        if filepaths is None:
            # Scan repo for python files
            filepaths = []
            for path in self.repo_path.rglob("*.py"):
                # Exclude virtual environments and hidden dirs
                parts = path.parts
                if any(p.startswith(".") or p in ("venv", ".venv", "env", "node_modules", "__pycache__", "build", "dist") for p in parts):
                    continue
                rel = path.relative_to(self.repo_path).as_posix()
                filepaths.append(rel)

        for fp in filepaths:
            normalized = fp.replace("\\", "/")
            results[normalized] = self.analyze_file(normalized)

        return results
