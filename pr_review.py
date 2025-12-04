import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


load_dotenv()

GITHUB_API_BASE = "https://api.github.com"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


class ConfigError(Exception):
    """Raised when required environment variables are missing."""
    pass


def debug(msg: str) -> None:
    """Simple STDOUT debug logger."""
    print(f"[DEBUG] {msg}", file=sys.stdout)


def get_env_var(name: str) -> str:
    """Fetch a required environment variable; fail fast if missing."""
    value = os.getenv(name)
    if not value:
        raise ConfigError(
            f"Environment variable '{name}' is required but not found in .env"
        )
    return value


def fetch_pr_details(repo: str, pr_number: int, github_token: str) -> Dict[str, Any]:
    """Fetch metadata of the Pull Request (title, author, etc.)."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    debug(f"Fetching PR details: {url}")
    resp = requests.get(url, headers=headers, timeout=30)

    if resp.status_code == 404:
        raise RuntimeError(f"PR #{pr_number} not found in repo {repo}.")

    resp.raise_for_status()
    return resp.json()


def fetch_pr_files(
    repo: str, pr_number: int, github_token: str
) -> List[Dict[str, Any]]:
    """
    Fetch all modified files in a PR.
    Handles pagination since large PRs may exceed 100 items/page.
    """
    files: List[Dict[str, Any]] = []
    page = 1
    per_page = 100

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }

    while True:
        url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/files?page={page}&per_page={per_page}"
        debug(f"Fetching PR files page {page}: {url}")

        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        page_items = resp.json()

        if not page_items:
            break

        files.extend(page_items)

        if len(page_items) < per_page:
            break  # no more pages

        page += 1

    return files


def build_changes_for_prompt(
    files: List[Dict[str, Any]], max_patch_chars: int = 8000
) -> List[Dict[str, str]]:
    """
    Prepare file diffs in a clean and size-controlled format suitable for LLM input.
    Large patches are truncated to avoid token explosion.
    """
    changes: List[Dict[str, str]] = []
    for f in files:
        filename = f.get("filename")
        status = f.get("status")
        patch = f.get("patch")

        if not patch:
            continue

        if len(patch) > max_patch_chars:
            patch = patch[:max_patch_chars] + "\n...TRUNCATED..."

        changes.append({"path": filename, "status": status, "patch": patch})

    return changes


def build_deepseek_messages(
    repo: str,
    pr_number: int,
    pr_title: str,
    pr_author: str,
    changes: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Construct the system + user messages for LLM-based PR review.
    Ensures the model outputs strict JSON as requested.
    """
    system = textwrap.dedent(
        """
        You are a senior software engineer reviewing PR diffs. 
        Respond ONLY with valid JSON — no extra text or explanations.
    """
    ).strip()

    instructions = {
        "task": "review_pull_request",
        "repo": repo,
        "pr_number": pr_number,
        "title": pr_title,
        "author": pr_author,
        "changes": changes,
        "schema": {
            "summary": "short text",
            "findings": [
                {
                    "file": "path",
                    "line_hint": "line number or approx",
                    "type": "bug | style | suggestion | performance | security | test",
                    "severity": "low | medium | high",
                    "message": "issue details",
                    "suggested_fix": "optional fix",
                }
            ],
        },
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(instructions, ensure_ascii=False)},
    ]


def call_deepseek_review(
    api_key: str, messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Execute the DeepSeek chat completion request.
    Enforces JSON output via response_format.
    """
    url = f"{DEEPSEEK_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    debug(f"Calling DeepSeek at {url}")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code == 401:
        raise RuntimeError("DeepSeek unauthorized. Check key in .env.")

    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def format_review_markdown(review: Dict[str, Any]) -> str:
    """
    Convert LLM JSON response into clean Markdown suitable as a GitHub PR review.
    """
    summary = review.get("summary", "")
    findings = review.get("findings", [])

    body = [f"## 🤖 DeepSeek Review", "", summary, ""]
    if not findings:
        body.append("_No issues found by AI reviewer._")
        return "\n".join(body)

    body.append("### Findings")
    for f in findings:
        body.append(
            f"- **{f['file']}** ({f.get('line_hint')}) — _{f['type']}/{f['severity']}_"
        )
        body.append(f"  - {f['message']}")

        if f.get("suggested_fix"):
            body.append("  - Suggested fix:")
            body.append("    ```")
            body.append(f"    {f['suggested_fix']}")
            body.append("    ```")

    return "\n".join(body)


def post_pr_review_comment(repo: str, pr_number: int, token: str, body: str):
    """Submit the final review as a GitHub Pull Request review comment."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}/reviews"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    resp = requests.post(
        url, headers=headers, json={"body": body, "event": "COMMENT"}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def main():
    """CLI entry point for running a full PR → LLM → GitHub review cycle."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo")
    parser.add_argument("--pr", required=True, type=int, help="PR number")

    args = parser.parse_args()
    repo, pr = args.repo, args.pr

    # Load credentials early so failures occur before network calls.
    token = get_env_var("GITHUB_TOKEN")
    deepseek = get_env_var("DEEPSEEK_API_KEY")

    debug(f"Reviewing PR #{pr} in {repo}")

    # Fetch PR metadata + files
    pr_data = fetch_pr_details(repo, pr, token)
    files = fetch_pr_files(repo, pr, token)

    # Convert to LLM-friendly format
    changes = build_changes_for_prompt(files)
    messages = build_deepseek_messages(
        repo, pr, pr_data["title"], pr_data["user"]["login"], changes
    )

    # Generate AI review
    review = call_deepseek_review(deepseek, messages)

    # Format & post back to GitHub
    body = format_review_markdown(review)
    resp = post_pr_review_comment(repo, pr, token, body)

    print("✔️ Review posted. ID:", resp.get("id"))


if __name__ == "__main__":
    main()
