#!/usr/bin/env python3
"""Sanitize CUDA entries in compile_commands.json so clang-based tooling
(clang-tidy, clangd) can replay them.

nvcc accepts several flags that its own clang-tidy/clang front-end does not
recognize (-ccbin, -forward-unknown-to-host-compiler, --extended-lambda,
--expt-relaxed-constexpr, -Xcompiler=...). Those are dropped outright, since
they carry no meaning for a syntax/semantic check. nvcc's own architecture
flag (--generate-code=arch=compute_XX,code=[...]) has no clang equivalent
either, but clang needs *some* target architecture, so it is translated to
clang's --cuda-gpu-arch=smXX rather than just discarded, or clang-tidy
silently falls back to its own default (sm_52) instead of the project's
actual target.

Operates by regex substitution directly on the raw "command" string rather
than tokenizing/rejoining it. Every flag handled here is a single token with
no embedded spaces, so this is sufficient, and it deliberately avoids
re-implementing shell tokenization: a prior version used shlex.split(),
which defaults to POSIX quoting rules and silently eats backslashes in
Windows paths (D:\\a\\cuda-demo\\... became D:acuda-demo...), corrupting
every Windows compile-command entry while leaving Linux ones untouched.
Regex substitution on the untouched string sidesteps needing to know which
platform's quoting rules apply at all.

Invoked by .github/actions/post-process-compile-commands as:
    post_process_compile_commands.py <source_db> <output_dir>
"""

import json
import re
import sys
from pathlib import Path

# Bodies of nvcc-only flags that plain clang does not recognize; dropped
# with no replacement. Each is a single token with no embedded spaces, so
# it is safe to match with a raw regex without tokenizing the command line.
_DROP_FLAG_BODIES = (
    r"-ccbin=[^\s\"]*",
    r"-forward-unknown-to-host-compiler",
    r"--extended-lambda",
    r"--expt-relaxed-constexpr",
)

# Matches each drop-flag body optionally wrapped in a single pair of quotes
# (some generators quote flags containing shell-special characters even on
# Windows) plus any leading whitespace, so removal doesn't leave a double
# space behind.
DROP_PATTERNS = [
    re.compile(rf'\s*"?{body}"?') for body in _DROP_FLAG_BODIES
]

# -Xcompiler has three forms nvcc accepts, and all three need to be dropped
# in full - a partial match (e.g. matching only "-Xcompiler=" and leaving a
# quoted, space-containing value behind as loose text) is what caused
# clang-cl to misread that leftover text as the value of an unrelated flag
# like /EH:
#   -Xcompiler=a,b,c          (single token, comma-joined, no spaces)
#   -Xcompiler="/EHsc /O2"    (single token, quoted value containing spaces)
#   -Xcompiler "/EHsc /O2"    (two tokens, quoted value containing spaces)
#   -Xcompiler /EHsc          (two tokens, single unquoted value)
DROP_PATTERNS.append(re.compile(r'\s*-Xcompiler=(?:"[^"]*"|[^\s"]*)'))
DROP_PATTERNS.append(re.compile(r'\s*-Xcompiler\s+"[^"]*"'))
DROP_PATTERNS.append(re.compile(r'\s*-Xcompiler\s+[^\s"]+'))
DROP_PATTERNS = tuple(DROP_PATTERNS)

# Locates an installed CUDA Toolkit's root directory from an -I/-isystem
# include path already present in the command (e.g. ...\CUDA\v13.3\include
# on Windows, or /usr/local/cuda-13.3/targets/.../include on Linux).
# clang-tidy's standalone clang front-end, unlike nvcc, has no reliable
# default search path for this on Windows, and errors with "cannot find
# CUDA installation" / "cannot find libdevice" without it.
_CUDA_ROOT_PATTERNS = (
    re.compile(r'[A-Za-z]:[\\/](?:[^"\\/]+[\\/])*CUDA[\\/]v[\d.]+'),
    re.compile(r"/[^\"\s]*?/cuda-[\d.]+"),
)

# nvcc's architecture flag, e.g.:
#   --generate-code=arch=compute_75,code=[compute_75,sm_75]
# translated to clang's:
#   --cuda-gpu-arch=sm_75
GENERATE_CODE_PATTERN = re.compile(
    r'"?--generate-code=arch=compute_(\d+),[^"\s]*"?'
)

# Token-exact equivalents of the above, for compile_commands.json entries
# that use an "arguments" list instead of a "command" string. No quoting to
# strip there, since each element is already a real argv token - a value
# with embedded spaces still arrives as a single list element, so the
# equals-form needs no quote handling here the way the command-string
# regex above does.
_DROP_TOKEN_PATTERNS = tuple(
    re.compile(rf"^{body}$") for body in (*_DROP_FLAG_BODIES, r"-Xcompiler=.*")
)
GENERATE_CODE_TOKEN_PATTERN = re.compile(
    r"^--generate-code=arch=compute_(\d+),.*$"
)


def find_cuda_root(command: str) -> str | None:
    for pattern in _CUDA_ROOT_PATTERNS:
        match = pattern.search(command)
        if match:
            return match.group(0)
    return None


def sanitize_command(command: str, is_cuda_file: bool) -> str:
    result = command
    for pattern in DROP_PATTERNS:
        result = pattern.sub("", result)
    result = GENERATE_CODE_PATTERN.sub(
        lambda m: f"--cuda-gpu-arch=sm_{m.group(1)}", result
    )

    if is_cuda_file:
        cuda_root = find_cuda_root(command)
        if cuda_root:
            result += f' --cuda-path="{cuda_root}"'

    # Collapse whitespace left behind by removals; do not otherwise alter
    # spacing, quoting, or backslashes anywhere else in the line.
    return re.sub(r"[ \t]{2,}", " ", result).strip()


def sanitize_tokens(tokens: list[str], is_cuda_file: bool) -> list[str]:
    sanitized: list[str] = []
    skip_next = False
    cuda_root: str | None = None

    for token in tokens:
        if skip_next:
            skip_next = False
            continue

        if token == "-Xcompiler":
            # Two-token form: -Xcompiler <value>; drop both.
            skip_next = True
            continue

        if any(pattern.match(token) for pattern in _DROP_TOKEN_PATTERNS):
            continue

        arch_match = GENERATE_CODE_TOKEN_PATTERN.match(token)
        if arch_match:
            sanitized.append(f"--cuda-gpu-arch=sm_{arch_match.group(1)}")
            continue

        if is_cuda_file and cuda_root is None:
            cuda_root = find_cuda_root(token)

        sanitized.append(token)

    if is_cuda_file and cuda_root:
        sanitized.append(f"--cuda-path={cuda_root}")

    return sanitized


def main() -> None:
    source_db = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_db = output_dir / "compile_commands.json"

    entries = json.loads(source_db.read_text(encoding="utf-8"))

    for entry in entries:
        # The "file" field names exactly what this entry compiles, which is
        # more reliable than pattern-matching the command line for ".cu" -
        # a path component could coincidentally contain that substring.
        is_cuda_file = str(entry.get("file", "")).endswith(".cu")

        if "command" in entry:
            entry["command"] = sanitize_command(entry["command"], is_cuda_file)
        elif "arguments" in entry:
            entry["arguments"] = sanitize_tokens(entry["arguments"], is_cuda_file)

    output_db.write_text(json.dumps(entries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
