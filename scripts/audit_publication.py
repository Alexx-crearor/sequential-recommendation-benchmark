"""Inspect staged Git blobs for large artifacts and likely secrets; never prints values."""
import argparse
import json
from pathlib import Path
import re
import subprocess

SECRET_PATTERNS = {
    "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "GitHub token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "AWS access key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "API bearer key": rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b",
    "literal credential": rb'''(?im)^\s*["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']''',
}
PLACEHOLDERS = (b"example", b"placeholder", b"your_", b"your-", b"replace", b"changeme")


def audit(root, max_bytes=5 * 1024 * 1024):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    files, findings = [], []
    for entry in filter(None, entries):
        meta, raw_path = entry.split(b"\t", 1)
        mode, oid, stage = meta.split()
        path = raw_path.decode("utf-8")
        if mode == b"160000":
            findings.append({"path": path, "reason": "nested Git repository/gitlink"})
            continue
        blob = git("cat-file", "blob", oid.decode())
        files.append({"path": path, "bytes": len(blob)})
        lower = path.lower()
        if len(blob) > max_bytes:
            findings.append({"path": path, "reason": "file exceeds publication size limit"})
        if lower.startswith(("data/", "references/")) or Path(lower).suffix in {".pt", ".pth", ".ckpt", ".parquet", ".pkl", ".npy", ".npz", ".zip", ".log"} or "evaluation_users" in lower or "predictions.csv" in lower:
            findings.append({"path": path, "reason": "dataset, checkpoint, prediction or intermediate artifact"})
        if Path(lower).name in {".env", "kaggle.json"} or Path(lower).suffix in {".pem", ".key", ".p12", ".pfx"}:
            findings.append({"path": path, "reason": "credential filename"})
        for label, pattern in SECRET_PATTERNS.items():
            for match in re.finditer(pattern, blob):
                value = match.group(1) if match.lastindex else match.group()
                if not any(marker in value.lower() for marker in PLACEHOLDERS):
                    findings.append({"path": path, "reason": label})
    return {"scope": "staged Git blobs", "file_count": len(files), "total_bytes": sum(f["bytes"] for f in files),
            "largest_files": sorted(files, key=lambda x: x["bytes"], reverse=True)[:10], "files": files, "findings": findings}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root, args.max_bytes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("scope", "file_count", "total_bytes", "findings")}, indent=2))
    raise SystemExit(bool(report["findings"]))
