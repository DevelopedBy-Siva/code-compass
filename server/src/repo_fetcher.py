import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".java",
    ".go",
    ".rs",
    ".md",
    ".mdx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".sh",
    ".css",
    ".html",
    ".prisma",
}

SUPPORTED_FILENAMES = {
    ".env.example",
    "Dockerfile",
}

IGNORED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "bun.lockb",
}

IGNORED_DIRS = {
    ".git",
    ".next",
    ".turbo",
    "dist",
    "build",
    "coverage",
    "node_modules",
    "vendor",
    ".venv",
    "venv",
    "__pycache__",
}

MAX_FILE_SIZE_BYTES = 250_000


class RepoFetcher:
    def __init__(self, base_dir: str = None):
        repo_cache_dir = base_dir or os.getenv(
            "REPO_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "codecompass-repos"),
        )
        self.base_dir = Path(repo_cache_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def parse_github_url(self, github_url: str) -> dict:
        parsed = urlparse(github_url)
        path = parsed.path.rstrip("/")
        if parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError("Only github.com URLs are supported")

        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub URL must include owner and repository name")

        owner = parts[0]
        repo = parts[1].removesuffix(".git")
        branch = "main"

        if len(parts) >= 4 and parts[2] in {"tree", "blob"}:
            branch = parts[3]

        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{owner}-{repo}")
        repo_url = f"https://github.com/{owner}/{repo}"
        return {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "slug": slug,
            "repo_url": repo_url,
        }

    def clone_repository(self, github_url: str) -> dict:
        info = self.parse_github_url(github_url)
        target_dir = self.base_dir / info["slug"]

        if target_dir.exists():
            shutil.rmtree(target_dir)

        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            info["branch"],
            github_url,
            str(target_dir),
        ]

        clone_cmd[6] = info["repo_url"]

        result = subprocess.run(clone_cmd, capture_output=True, text=True)
        if result.returncode != 0 and info["branch"] != "main":
            info["branch"] = "main"
            clone_cmd[5] = "main"
            result = subprocess.run(clone_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            default_branch = self._resolve_default_branch(info["repo_url"])
            if default_branch and default_branch != info["branch"]:
                info["branch"] = default_branch
                clone_cmd[5] = default_branch
                result = subprocess.run(clone_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Failed to clone repository")

        return {
            **info,
            "local_path": str(target_dir),
        }

    def _resolve_default_branch(self, github_url: str) -> str | None:
        result = subprocess.run(
            ["git", "ls-remote", "--symref", github_url, "HEAD"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        for line in result.stdout.splitlines():
            if line.startswith("ref: ") and "\tHEAD" in line:
                ref = line.split("\t", 1)[0].removeprefix("ref: ").strip()
                if ref.startswith("refs/heads/"):
                    return ref.removeprefix("refs/heads/")
        return None

    def cleanup_repository(self, repo_path: str):
        target = Path(repo_path)
        if target.exists():
            shutil.rmtree(target)

    def iter_source_files(self, repo_path: str):
        root = Path(repo_path)
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in IGNORED_DIRS for part in file_path.parts):
                continue
            if file_path.name in IGNORED_FILENAMES:
                continue
            if (
                file_path.suffix.lower() not in SUPPORTED_EXTENSIONS
                and file_path.name not in SUPPORTED_FILENAMES
            ):
                continue
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            yield file_path
