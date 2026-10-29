#!/usr/bin/env python3
"""Scan this Mac for disk space that can be freed and write a Markdown report.

The scanner itself does not delete files. It also writes a zsh or bash script
that shows each reclaimable section and deletes that section only after a yes.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DROP_SEGMENTS = {
    "gguf",
    "ggml",
    "mlx",
    "ckpt",
    "safetensors",
    "bin",
    "f16",
    "f32",
    "bf16",
    "fp16",
    "fp32",
    "qat",
    "it",
    "m5air",
    "mtp",
    "i1",
    "turboquant",
    "rotorquant",
    "mxfp4",
    "mxfp8",
    "optiq",
}

QUANT_LEFTOVERS = {"k", "m", "s", "xs", "l", "xl"}

BASE_PATTERNS = [
    re.compile(r"qwen\d+(?:\.\d+)?-\d+b(?:-a\d+b)?"),
    re.compile(r"gpt-oss-\d+b"),
    re.compile(r"gemma-\d+(?:\.\d+)?-e\d+b"),
    re.compile(r"gemma-\d+(?:\.\d+)?-\d+b-a\d+b"),
    re.compile(r"gemma-\d+(?:\.\d+)?-moongem-\d+b"),
    re.compile(r"minicpm\d+-\d+b"),
    re.compile(r"granite-[\d.]+-\d+b"),
]

PARTIAL_SUFFIXES = (".partial", ".crdownload", ".download", ".part")
INSTALLER_SUFFIXES = {".dmg", ".pkg"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv"}
AUDIO_SUFFIXES = {".wav", ".aif", ".aiff", ".flac"}


@dataclass
class SizedPath:
    path: Path
    size: int
    note: str = ""


@dataclass
class ModelCopy:
    path: Path
    size: int
    exact_key: str
    base_key: str


@dataclass
class DeleteAction:
    path: Path
    size: int
    kind: str


@dataclass
class DeleteSection:
    title: str
    summary: str
    lines: list[str]
    actions: list[DeleteAction]

    @property
    def size(self) -> int:
        return sum(action.size for action in self.actions)


# Exact paths the delete script must never remove. Children of these paths can still be removed.
DENIED_RELATIVE = (
    Path("Library"),
    Path("Library/Application Support"),
    Path("Library/Application Support/Evernote"),
    Path("Library/Application Support/Logos4"),
    Path("Library/Application Support/Bitwig"),
    Path("Documents"),
    Path("Documents/Bitwig Studio"),
    Path("Downloads"),
    Path("projects"),
    Path("projects/Obsidian Backups"),
    Path("Pictures"),
    Path("Pictures/Photos Library.photoslibrary"),
    Path(".lmstudio"),
    Path(".lmstudio/models"),
    Path(".cache"),
    Path(".hermes"),
)


@dataclass
class Scan:
    data_total: int
    data_used: int
    data_free: int
    home_folders: list[SizedPath] = field(default_factory=list)
    safe_caches: list[SizedPath] = field(default_factory=list)
    cache_breakdown: list[SizedPath] = field(default_factory=list)
    evernote: list[SizedPath] = field(default_factory=list)
    partials: list[SizedPath] = field(default_factory=list)
    models: list[ModelCopy] = field(default_factory=list)
    models_total: int = 0
    biglybt: list[SizedPath] = field(default_factory=list)
    biglybt_total: int = 0
    large_media: list[SizedPath] = field(default_factory=list)
    installers: list[SizedPath] = field(default_factory=list)
    other_downloads: list[SizedPath] = field(default_factory=list)
    libraries: list[SizedPath] = field(default_factory=list)
    other_caches: list[SizedPath] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def fmt_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def display_path(path: Path, home: Path) -> str:
    try:
        return "~/" + path.resolve().relative_to(home.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def parse_du_line(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    if not stripped:
        return None
    if "\t" in stripped:
        size_text, path_text = stripped.split("\t", 1)
    else:
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            return None
        size_text, path_text = parts
    try:
        return int(size_text) * 1024, path_text
    except ValueError:
        return None


def du_lines(path: Path, command: list[str], timeout: int) -> list[tuple[Path, int]]:
    if not path.exists():
        return []
    try:
        completed = run(command, timeout)
    except subprocess.TimeoutExpired:
        log(f"timed out measuring {path}")
        return []
    if not completed.stdout.strip():
        detail = completed.stderr.strip().splitlines()
        if detail:
            log(f"could not measure {path}: {detail[0]}")
        return []
    found: list[tuple[Path, int]] = []
    for line in completed.stdout.splitlines():
        parsed = parse_du_line(line)
        if parsed is None:
            continue
        size, text = parsed
        found.append((Path(text), size))
    return found


def du_tree(path: Path, max_depth: int, timeout: int = 180) -> list[tuple[Path, int]]:
    # -s and -d cannot be combined. -k selects 1024-byte blocks.
    return du_lines(path, ["du", "-k", "-d", str(max_depth), str(path)], timeout)


def du_one(path: Path, timeout: int = 180) -> int | None:
    rows = du_lines(path, ["du", "-sk", str(path)], timeout)
    if not rows:
        return None
    return rows[0][1]


def df_bytes(path: Path) -> tuple[int, int, int] | None:
    if not path.exists():
        return None
    try:
        completed = run(["df", "-k", str(path)], timeout=30)
    except subprocess.TimeoutExpired:
        return None
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    fields = lines[-1].split()
    if len(fields) < 4:
        return None
    try:
        total, used, free = (int(fields[1]) * 1024, int(fields[2]) * 1024, int(fields[3]) * 1024)
    except ValueError:
        usage = shutil.disk_usage(path)
        return usage.total, usage.used, usage.free
    return total, used, free


def is_quant_segment(segment: str) -> bool:
    return bool(
        re.fullmatch(r"q\d+[a-z0-9]*", segment)
        or re.fullmatch(r"iq\d+[a-z0-9]*", segment)
        or re.fullmatch(r"\d+bit", segment)
        or re.fullmatch(r"i\d+x", segment)
        or re.fullmatch(r"oq\d+e?", segment)
    )


def normalize_model_name(name: str) -> str:
    """Drop quant and file-format tokens so copies of one model share a key."""
    text = name.lower().replace("_", "-").replace(" ", "-")
    text = re.sub(r"\.(ckpt|gguf|safetensors|bin)$", "", text)
    kept: list[str] = []
    previous_was_quant = False
    for segment in re.split(r"-+", text):
        if not segment:
            continue
        quant = segment in DROP_SEGMENTS or is_quant_segment(segment)
        if quant:
            previous_was_quant = True
            continue
        if previous_was_quant and segment in QUANT_LEFTOVERS:
            continue
        previous_was_quant = False
        kept.append(segment)
    return "-".join(kept)


def base_key(exact_key: str) -> str:
    for pattern in BASE_PATTERNS:
        match = pattern.match(exact_key)
        if match:
            return match.group(0)
    return exact_key


def partial_note(path: Path) -> str:
    map_path = Path(str(path) + ".map")
    apparent = path.stat().st_size if path.exists() else 0
    allocated = du_one(path, timeout=60) or apparent
    pieces = []
    if apparent > int(allocated * 1.1):
        pieces.append(f"{fmt_bytes(apparent)} logical, {fmt_bytes(allocated)} stored")
    if map_path.is_file():
        try:
            payload = plistlib.loads(map_path.read_bytes())
        except (plistlib.InvalidFileException, OSError, ValueError):
            payload = None
        blocks = payload.get("completedBlocks") if isinstance(payload, dict) else None
        if isinstance(blocks, list) and blocks:
            done = sum(1 for block in blocks if block)
            pieces.append(f"{done} of {len(blocks)} blocks downloaded")
        expected = payload.get("expectedLength") if isinstance(payload, dict) else None
        if isinstance(expected, int) and expected:
            pieces.append(f"full file would be {fmt_bytes(expected)}")
    if not pieces:
        pieces.append("unfinished download")
    return "; ".join(pieces)


def find_files(root: Path, min_bytes: int, suffixes: set[str] | None = None, limit: int = 40) -> list[SizedPath]:
    if not root.exists():
        return []
    command = ["find", str(root), "-type", "f"]
    if suffixes:
        tests: list[str] = []
        for suffix in sorted(suffixes):
            if tests:
                tests.append("-o")
            tests.extend(["-iname", f"*{suffix}"])
        command.extend(["(", *tests, ")"])
    megabytes = max(1, min_bytes // (1024 * 1024))
    command.extend(["-size", f"+{megabytes}M"])
    try:
        completed = run(command, timeout=180)
    except subprocess.TimeoutExpired:
        log(f"timed out searching {root}")
        return []
    found: list[SizedPath] = []
    for line in completed.stdout.splitlines():
        path = Path(line)
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_bytes:
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        found.append(SizedPath(path, size))
        if len(found) >= limit:
            break
    found.sort(key=lambda item: item.size, reverse=True)
    return found


def child_sizes(path: Path, minimum: int, timeout: int = 180) -> tuple[int, list[SizedPath]]:
    rows = du_tree(path, 1, timeout)
    total = 0
    children: list[SizedPath] = []
    for child, size in rows:
        if child == path:
            total = size
            continue
        if size >= minimum:
            children.append(SizedPath(child, size))
    children.sort(key=lambda item: item.size, reverse=True)
    if total == 0 and rows:
        total = max(size for _, size in rows)
    return total, children


def collect_models(models_root: Path) -> tuple[int, list[ModelCopy]]:
    rows = du_tree(models_root, 2, timeout=240)
    by_depth: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for path, size in rows:
        try:
            depth = len(path.relative_to(models_root).parts)
        except ValueError:
            continue
        by_depth[depth].append((path, size))
    total = next((size for path, size in rows if path == models_root), 0)
    copies: list[ModelCopy] = []
    model_rows = by_depth.get(2, [])
    if not model_rows:
        model_rows = by_depth.get(1, [])
    for path, size in model_rows:
        if size < 100 * 1024 * 1024:
            continue
        exact = normalize_model_name(path.name)
        if not exact:
            exact = path.name.lower()
        copies.append(ModelCopy(path, size, exact, base_key(exact)))
    copies.sort(key=lambda item: item.size, reverse=True)
    return total, copies


def collect_partials(home: Path) -> list[SizedPath]:
    roots = [
        home / "Library/Containers/com.liuliu.draw-things/Data/Documents/Models",
        home / "Documents/Draw Things Models",
        home / ".lmstudio/models",
        home / "Downloads",
    ]
    found: list[SizedPath] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        command = ["find", str(root), "-type", "f", "("]
        for index, suffix in enumerate(PARTIAL_SUFFIXES):
            if index:
                command.append("-o")
            command.extend(["-name", f"*{suffix}"])
        command.append(")")
        try:
            completed = run(command, timeout=120)
        except subprocess.TimeoutExpired:
            log(f"timed out looking for partial downloads in {root}")
            continue
        for line in completed.stdout.splitlines():
            path = Path(line)
            if path.suffix == ".map" or path.name.endswith(".partial.map"):
                continue
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            allocated = du_one(path, timeout=60)
            apparent = path.stat().st_size
            size = allocated if allocated is not None else apparent
            found.append(SizedPath(path, size, partial_note(path)))
    found.sort(key=lambda item: item.size, reverse=True)
    return found


def scan(home: Path) -> Scan:
    data_path = Path("/System/Volumes/Data")
    disk = df_bytes(data_path if data_path.exists() else home)
    if disk is None:
        usage = shutil.disk_usage(home)
        disk = (usage.total, usage.used, usage.free)
    report = Scan(*disk)

    log("measuring home folders")
    _, home_children = child_sizes(home, 500 * 1024 * 1024, timeout=240)
    report.home_folders = home_children[:20]

    log("measuring caches")
    cache_roots = [
        (home / "Library/Caches", "App caches. Safe to remove. Apps rebuild them."),
        (home / ".npm", "npm and npx cache. Safe to remove."),
        (home / ".cache/uv", "uv package cache. Safe to remove."),
        (home / ".cache/pip", "pip cache. Safe to remove."),
        (home / ".lmstudio/server-logs", "LM Studio server logs. Safe to remove."),
    ]
    for path, note in cache_roots:
        size = du_one(path)
        if size and size >= 20 * 1024 * 1024:
            report.safe_caches.append(SizedPath(path, size, note))
    cache_total, cache_children = child_sizes(home / "Library/Caches", 100 * 1024 * 1024)
    report.cache_breakdown = cache_children[:15]
    if cache_total and not any(item.path == home / "Library/Caches" for item in report.safe_caches):
        report.safe_caches.append(
            SizedPath(home / "Library/Caches", cache_total, "App caches. Safe to remove. Apps rebuild them.")
        )

    _, other_cache_children = child_sizes(home / ".cache", 100 * 1024 * 1024)
    safe_cache_names = {"uv", "pip", "outlines", "opencode", "yt-dlp", "selenium"}
    for child in other_cache_children:
        if child.path.name == "huggingface":
            child.note = "Hugging Face cache. This can be model weights, so review it before deleting."
            report.other_caches.append(child)
        elif child.path.name not in safe_cache_names:
            child.note = "Tool cache. Usually safe, but check the folder name first."
            report.other_caches.append(child)

    log("measuring Evernote")
    evernote = home / "Library/Application Support/Evernote"
    evernote_names = {
        "resource-cache": "Downloaded note attachments. Evernote fetches them again when you open the note.",
        "Cache": "Evernote cache. Safe to remove.",
    }
    _, evernote_children = child_sizes(evernote, 50 * 1024 * 1024)
    for child in evernote_children:
        if child.path.name in evernote_names:
            child.note = evernote_names[child.path.name]
            report.evernote.append(child)

    log("looking for unfinished downloads")
    report.partials = collect_partials(home)

    log("measuring LM Studio models")
    models_root = home / ".lmstudio/models"
    if models_root.exists():
        report.models_total, report.models = collect_models(models_root)

    log("measuring downloads and large personal files")
    bigly = home / "Documents/BiglyBT Downloads"
    if bigly.exists():
        report.biglybt_total, big_children = child_sizes(bigly, 500 * 1024 * 1024, timeout=240)
        report.biglybt = big_children[:20]

    media_roots = [
        (home / "Documents/Bitwig Studio", AUDIO_SUFFIXES | VIDEO_SUFFIXES, 1024 * 1024 * 1024, "Large recording."),
        (home / "projects/Obsidian Backups", VIDEO_SUFFIXES | AUDIO_SUFFIXES, 500 * 1024 * 1024, "Large file inside an Obsidian backup."),
    ]
    for root, suffixes, minimum, note in media_roots:
        for item in find_files(root, minimum, suffixes, limit=25):
            item.note = note
            report.large_media.append(item)
    report.large_media.sort(key=lambda item: item.size, reverse=True)

    downloads = home / "Downloads"
    for item in find_files(downloads, 100 * 1024 * 1024, limit=40):
        if item.path.suffix.lower() in INSTALLER_SUFFIXES:
            item.note = "Installer. Safe to remove once that app is already installed."
            report.installers.append(item)
        elif item.path.suffix.lower() in PARTIAL_SUFFIXES:
            continue
        else:
            item.note = "Download. Review it before deleting."
            report.other_downloads.append(item)

    library_candidates = [
        (home / "Pictures/Photos Library.photoslibrary", "Photos library. This is your photo library, not a cache."),
        (home / "Library/Application Support/Logos4", "Logos library. This is app content, not a cache."),
        (home / "Library/Application Support/Bitwig", "Bitwig support files. Review before deleting."),
    ]
    for path, note in library_candidates:
        size = du_one(path, timeout=120)
        if size and size >= 1024 * 1024 * 1024:
            report.libraries.append(SizedPath(path, size, note))

    vm = df_bytes(Path("/System/Volumes/VM"))
    if vm and vm[1] >= 1024 * 1024 * 1024:
        report.notes.append(
            f"The swap volume is using {fmt_bytes(vm[1])}. A restart can release it."
        )
    return report


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def path_rows(items: list[SizedPath], home: Path, with_note: bool = True) -> list[list[str]]:
    rows = []
    for item in items:
        row = [f"`{display_path(item.path, home)}`", fmt_bytes(item.size)]
        if with_note:
            row.append(item.note)
        rows.append(row)
    return rows


def repeated_groups(models: list[ModelCopy]) -> list[tuple[str, list[ModelCopy]]]:
    grouped: dict[str, list[ModelCopy]] = defaultdict(list)
    for model in models:
        grouped[model.exact_key].append(model)
    repeats = [(key, copies) for key, copies in grouped.items() if len(copies) > 1]
    repeats.sort(key=lambda item: sum(copy.size for copy in item[1]), reverse=True)
    return repeats


def variant_groups(models: list[ModelCopy]) -> list[tuple[str, list[tuple[str, list[ModelCopy]]]]]:
    by_base: dict[str, dict[str, list[ModelCopy]]] = defaultdict(lambda: defaultdict(list))
    for model in models:
        by_base[model.base_key][model.exact_key].append(model)
    groups = []
    for base, exacts in by_base.items():
        if len(exacts) < 2:
            continue
        exact_rows = list(exacts.items())
        exact_rows.sort(key=lambda item: sum(copy.size for copy in item[1]), reverse=True)
        groups.append((base, exact_rows))
    groups.sort(
        key=lambda item: sum(copy.size for _, copies in item[1] for copy in copies),
        reverse=True,
    )
    return groups


def repeat_savings(groups: list[tuple[str, list[ModelCopy]]]) -> int:
    saved = 0
    for _, copies in groups:
        saved += sum(copy.size for copy in copies) - max(copy.size for copy in copies)
    return saved


def variant_extra_savings(groups: list[tuple[str, list[tuple[str, list[ModelCopy]]]]]) -> int:
    """Space freed by keeping only the largest file in a base family, beyond exact-copy savings."""
    extra = 0
    for _, exacts in groups:
        largest_of_each = []
        for _, copies in exacts:
            largest_of_each.append(max(copy.size for copy in copies))
        extra += sum(largest_of_each) - max(largest_of_each)
    return extra


def is_deletable(path: Path, home: Path, kind: str) -> bool:
    if kind not in {"file", "dir"}:
        return False
    if ".." in path.parts:
        return False
    try:
        resolved = path.resolve()
        home_resolved = home.resolve()
    except OSError:
        return False
    if resolved == home_resolved or resolved == Path("/"):
        return False
    try:
        relative = resolved.relative_to(home_resolved)
    except ValueError:
        return False
    if relative in DENIED_RELATIVE:
        return False
    if resolved.exists():
        if kind == "file" and not resolved.is_file():
            return False
        if kind == "dir" and not resolved.is_dir():
            return False
    return True


def accepted_actions(actions: list[DeleteAction], home: Path) -> list[DeleteAction]:
    kept: list[DeleteAction] = []
    for action in actions:
        if is_deletable(action.path, home, action.kind):
            kept.append(action)
        else:
            log(f"not offering to delete {action.path}")
    return kept


def section_if_any(
    title: str,
    summary: str,
    lines: list[str],
    actions: list[DeleteAction],
    home: Path,
) -> DeleteSection | None:
    safe = accepted_actions(actions, home)
    if not safe:
        return None
    return DeleteSection(title, summary, lines, safe)


def build_delete_sections(report: Scan, home: Path) -> list[DeleteSection]:
    sections: list[DeleteSection] = []

    if report.safe_caches:
        lines = []
        actions = []
        for item in report.safe_caches:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
            if item.note:
                lines.append(f"            {item.note}")
            actions.append(DeleteAction(item.path, item.size, "dir"))
        if report.cache_breakdown:
            lines.append("")
            lines.append("Largest folders inside ~/Library/Caches, removed with that folder:")
            for item in report.cache_breakdown:
                lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
        found = section_if_any(
            "Safe caches",
            "Apps rebuild these folders.",
            lines,
            actions,
            home,
        )
        if found:
            sections.append(found)

    if report.evernote:
        lines = []
        actions = []
        for item in report.evernote:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
            if item.note:
                lines.append(f"            {item.note}")
            actions.append(DeleteAction(item.path, item.size, "dir"))
        found = section_if_any(
            "Evernote cache",
            "Evernote downloads these attachments again when you open the note. The notebook database is not in this section.",
            lines,
            actions,
            home,
        )
        if found:
            sections.append(found)

    if report.partials:
        lines = []
        actions = []
        for item in report.partials:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
            if item.note:
                lines.append(f"            {item.note}")
            actions.append(DeleteAction(item.path, item.size, "file"))
            map_path = Path(str(item.path) + ".map")
            if map_path.is_file():
                actions.append(DeleteAction(map_path, map_path.stat().st_size, "file"))
                lines.append(f"{fmt_bytes(map_path.stat().st_size):>10}  {display_path(map_path, home)}")
        found = section_if_any(
            "Unfinished downloads",
            "Incomplete files. Finished models are not in this section.",
            lines,
            actions,
            home,
        )
        if found:
            sections.append(found)

    repeat_lines: list[str] = []
    repeat_actions: list[DeleteAction] = []
    for key, copies in repeated_groups(report.models):
        ordered = sorted(copies, key=lambda copy: copy.size, reverse=True)
        keep = ordered[0]
        repeat_lines.append(f"{key}: keep {fmt_bytes(keep.size)}  {display_path(keep.path, home)}")
        for copy in ordered[1:]:
            repeat_lines.append(f"  delete {fmt_bytes(copy.size):>10}  {display_path(copy.path, home)}")
            repeat_actions.append(DeleteAction(copy.path, copy.size, "dir"))
        repeat_lines.append("")
    found = section_if_any(
        "Repeated model files",
        "Deletes extra copies of the same model name. The largest copy of each name stays.",
        repeat_lines,
        repeat_actions,
        home,
    )
    if found:
        sections.append(found)

    variant_lines: list[str] = []
    variant_actions: list[DeleteAction] = []
    for base, exacts in variant_groups(report.models):
        copies = [copy for _, group in exacts for copy in group]
        largest = max(copies, key=lambda copy: copy.size)
        variant_lines.append(f"{base}: keep variant `{largest.exact_key}`")
        for exact, group in exacts:
            ordered = sorted(group, key=lambda copy: copy.size, reverse=True)
            if exact == largest.exact_key:
                for copy in ordered:
                    variant_lines.append(f"  keep   {fmt_bytes(copy.size):>10}  {display_path(copy.path, home)}")
                continue
            for copy in ordered:
                variant_lines.append(f"  delete {fmt_bytes(copy.size):>10}  {display_path(copy.path, home)}")
                variant_actions.append(DeleteAction(copy.path, copy.size, "dir"))
        variant_lines.append("")
    found = section_if_any(
        "Other variants of the same base model",
        "Deletes other finetunes and builds. The variant that contains the largest file stays, including its extra copies.",
        variant_lines,
        variant_actions,
        home,
    )
    if found:
        sections.append(found)

    bigly = home / "Documents" / "BiglyBT Downloads"
    if report.biglybt_total and bigly.exists():
        lines = [
            f"{fmt_bytes(report.biglybt_total):>10}  {display_path(bigly, home)}",
            "",
            "Largest items inside that folder:",
        ]
        for item in report.biglybt:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
        lines.append("")
        lines.append("Confirming deletes the whole folder, including items not listed above.")
        found = section_if_any(
            "BiglyBT downloads",
            "Downloaded files in the BiglyBT folder.",
            lines,
            [DeleteAction(bigly, report.biglybt_total, "dir")],
            home,
        )
        if found:
            sections.append(found)

    def file_section(title: str, summary: str, items: list[SizedPath]) -> None:
        if not items:
            return
        lines = []
        actions = []
        for item in items:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
            if item.note:
                lines.append(f"            {item.note}")
            actions.append(DeleteAction(item.path, item.size, "file"))
        found_section = section_if_any(title, summary, lines, actions, home)
        if found_section:
            sections.append(found_section)

    file_section(
        "Large recordings and backup media",
        "Large audio and video files. Other project files stay.",
        report.large_media,
    )
    file_section(
        "Installer downloads",
        "Disk images and installer packages. Safe once that app is already installed.",
        report.installers,
    )
    file_section(
        "Other large downloads",
        "Other large files in Downloads. These are your files.",
        report.other_downloads,
    )

    if report.other_caches:
        lines = []
        actions = []
        for item in report.other_caches:
            lines.append(f"{fmt_bytes(item.size):>10}  {display_path(item.path, home)}")
            if item.note:
                lines.append(f"            {item.note}")
            actions.append(DeleteAction(item.path, item.size, "dir"))
        found = section_if_any(
            "Other caches to check",
            "Tool caches that may hold downloaded data.",
            lines,
            actions,
            home,
        )
        if found:
            sections.append(found)

    return sections


TERMINAL_MARKERS = (
    "terminal",
    "iterm",
    "ghostty",
    "kitty",
    "alacritty",
    "wezterm",
    "warp",
    "hyper",
)


def shell_from_comm(comm: str) -> str | None:
    """Return bash or zsh when a process command is one of those shells."""
    name = Path(comm.strip()).name
    if name.startswith("-"):
        name = name[1:]
    if name in {"zsh", "bash"}:
        return name
    return None


def is_terminal_app(comm: str) -> bool:
    text = comm.lower()
    return any(marker in text for marker in TERMINAL_MARKERS)


def choose_shell(chain: list[str]) -> tuple[str, str] | None:
    """Pick the shell the terminal is running.

    `chain` starts at the current process and walks toward the root.
    The shell next to the terminal app is the terminal shell. An inner
    bash or zsh used only to launch this scan does not override it.
    """
    shells = [
        (index, shell, chain[index])
        for index, shell in (
            (index, shell_from_comm(comm)) for index, comm in enumerate(chain)
        )
        if shell is not None
    ]
    terminal_at = next((index for index, comm in enumerate(chain) if is_terminal_app(comm)), None)
    if terminal_at is not None:
        attached = [item for item in shells if item[0] < terminal_at]
        if attached:
            _, shell, comm = attached[-1]
            app = Path(chain[terminal_at]).name
            return shell, f"the shell attached to {app} ({comm})"
    login_shells = [item for item in shells if Path(item[2]).name.startswith("-")]
    if login_shells:
        _, shell, comm = login_shells[-1]
        return shell, f"the login shell in this session ({comm})"
    if shells:
        _, shell, comm = shells[0]
        return shell, f"the shell that launched this scan ({comm})"
    return None


def process_chain() -> list[str]:
    """Command names from this process up to launchd or pid 1."""
    chain: list[str] = []
    pid = os.getpid()
    seen: set[int] = set()
    for _ in range(30):
        if pid <= 0 or pid in seen:
            break
        seen.add(pid)
        try:
            completed = run(["ps", "-o", "ppid=,comm=", "-p", str(pid)], timeout=5)
        except subprocess.TimeoutExpired:
            break
        line = completed.stdout.strip()
        if not line:
            break
        parts = line.split(None, 1)
        if len(parts) != 2:
            break
        try:
            parent = int(parts[0])
        except ValueError:
            break
        chain.append(parts[1].strip())
        if parent == pid:
            break
        pid = parent
    return chain


def detect_shell(explicit: str | None) -> tuple[str, str]:
    """Return the shell language and a short description of how it was chosen."""
    if explicit in {"zsh", "bash"}:
        return explicit, "chosen with --shell"
    chosen = choose_shell(process_chain())
    if chosen is not None:
        return chosen
    shell_env = os.environ.get("SHELL", "")
    name = Path(shell_env).name
    if name in {"zsh", "bash"}:
        return name, f"the account shell in $SHELL ({shell_env})"
    if Path("/bin/zsh").exists():
        return "zsh", "/bin/zsh, because no terminal shell could be identified"
    return "bash", "/bin/bash, because no terminal shell could be identified"


def default_script_name(shell: str) -> str:
    return f"delete-disk-space.{shell}"


def render_delete_script(
    sections: list[DeleteSection],
    home: Path,
    shell: str,
    generated_at: str,
    detection: str = "",
) -> str:
    home_text = str(home.resolve())
    denied = [home_text, "/"]
    denied.extend(str((home / relative).resolve()) for relative in DENIED_RELATIVE)
    detected = f"# Shell language: {shell}."
    if detection:
        detected += f" Detected from {detection}."
    if shell == "zsh":
        header = [
            "#!/bin/zsh",
            "# Generated by disk_space_report.py. Running it asks before each section.",
            f"# Generated {generated_at} for {home_text}.",
            detected,
            "emulate -L zsh",
            "setopt LOCAL_OPTIONS NO_UNSET",
            "",
        ]
        confirm_fn = [
            "confirm_section() {",
            "  local answer",
            "  print -n -r -- \"Delete this section? [y/N] \"",
            "  if ! read -r answer; then",
            "    print -r -- \"\"",
            "    return 1",
            "  fi",
            '  [[ "${answer:l}" == "y" || "${answer:l}" == "yes" ]]',
            "}",
            "",
        ]
        say = "print -r --"
    else:
        header = [
            "#!/bin/bash",
            "# Generated by disk_space_report.py. Running it asks before each section.",
            f"# Generated {generated_at} for {home_text}.",
            detected,
            "set -u",
            "",
        ]
        confirm_fn = [
            "confirm_section() {",
            "  local answer lower",
            "  printf '%s' \"Delete this section? [y/N] \"",
            "  if ! read -r answer; then",
            "    printf '\\n'",
            "    return 1",
            "  fi",
            "  lower=$(printf '%s' \"$answer\" | tr '[:upper:]' '[:lower:]')",
            '  [[ "$lower" == "y" || "$lower" == "yes" ]]',
            "}",
            "",
        ]
        say = "printf '%s\\n'"

    lines = list(header)
    lines.append(f"HOME_DIR={shlex.quote(home_text)}")
    lines.append("DENY=(")
    for path in denied:
        lines.append(f"  {shlex.quote(path)}")
    lines.append(")")
    lines.append("")
    lines.extend(
        [
            "refuse() {",
            f"  {say} \"Refusing to delete: $1\"",
            "  return 1",
            "}",
            "",
            "safe_to_remove() {",
            '  local target="$1"',
            '  local kind="$2"',
            '  local denied',
            '  [[ "$target" == /* ]] || { refuse "$target"; return 1; }',
            '  [[ "$target" != *..* ]] || { refuse "$target"; return 1; }',
            '  [[ "$target" == "$HOME_DIR/"* ]] || { refuse "$target"; return 1; }',
            '  for denied in "${DENY[@]}"; do',
            '    if [[ "$target" == "$denied" ]]; then',
            '      refuse "$target"',
            "      return 1",
            "    fi",
            "  done",
            '  if [[ "$kind" == "file" ]]; then',
            '    if [[ -e "$target" && ! -f "$target" ]]; then',
            '      refuse "$target"',
            "      return 1",
            "    fi",
            '  elif [[ "$kind" == "dir" ]]; then',
            '    if [[ -e "$target" && ! -d "$target" ]]; then',
            '      refuse "$target"',
            "      return 1",
            "    fi",
            "  else",
            '    refuse "$target"',
            "    return 1",
            "  fi",
            "  return 0",
            "}",
            "",
            "remove_path() {",
            '  local target="$1"',
            '  local kind="$2"',
            '  safe_to_remove "$target" "$kind" || return 1',
            '  if [[ "$kind" == "dir" ]]; then',
            '    rm -rf -- "$target" || return 1',
            "  else",
            '    rm -f -- "$target" || return 1',
            "  fi",
            f"  {say} \"  removed: $target\"",
            "}",
            "",
            "remove_counted() {",
            '  local target="$1"',
            '  local kind="$2"',
            '  local bytes="$3"',
            '  if grep -F -x -q -- "$target" "$REMOVED_LIST"; then',
            f"    {say} \"  already removed in an earlier section: $target\"",
            "    return 0",
            "  fi",
            '  if [[ ! -e "$target" && ! -L "$target" ]]; then',
            f"    {say} \"  already gone: $target\"",
            "    printf '%s\\n' \"$target\" >> \"$REMOVED_LIST\"",
            "    return 0",
            "  fi",
            '  remove_path "$target" "$kind" || return 1',
            "  printf '%s\\n' \"$target\" >> \"$REMOVED_LIST\"",
            "  deleted_bytes=$((deleted_bytes + bytes))",
            "}",
            "",
            "format_bytes() {",
            "  awk -v bytes=\"$1\" 'BEGIN {",
            '    split("B KB MB GB TB", units, " ")',
            "    value = bytes + 0",
            "    unit = 1",
            "    while (value >= 1024 && unit < 5) { value /= 1024; unit++ }",
            '    if (unit == 1) printf "%d %s\\n", value, units[unit]',
            '    else printf "%.1f %s\\n", value, units[unit]',
            "  }'",
            "}",
            "",
        ]
    )
    lines.extend(confirm_fn)
    lines.extend(
        [
            "confirmed=0",
            "skipped=0",
            "deleted_bytes=0",
            'REMOVED_LIST=$(mktemp)',
            'trap \'rm -f "$REMOVED_LIST"\' EXIT',
            f"{say} \"Disk space cleanup\"",
            f"{say} \"Nothing is deleted unless you answer y or yes for that section.\"",
            f"{say} \"Press Enter to skip a section.\"",
            f"{say} \"\"",
        ]
    )
    if not sections:
        lines.append(f"{say} \"Nothing reclaimable was found.\"")
        lines.append("exit 0")
        return "\n".join(lines).rstrip() + "\n"

    for section in sections:
        lines.append(f"{say} \"----------------------------------------------------------------\"")
        lines.append(f"{say} {shlex.quote('Section: ' + section.title)}")
        lines.append(f"{say} {shlex.quote('Reclaimable: ' + fmt_bytes(section.size))}")
        lines.append(f"{say} {shlex.quote(section.summary)}")
        lines.append(f"{say} \"\"")
        for report_line in section.lines:
            if report_line == "":
                lines.append(f"{say} \"\"")
            else:
                lines.append(f"{say} {shlex.quote(report_line)}")
        lines.append(f"{say} \"\"")
        lines.append("if confirm_section; then")
        lines.append("  remove_failed=0")
        for action in section.actions:
            quoted_path = shlex.quote(str(action.path.resolve()))
            lines.append(
                f"  remove_counted {quoted_path} {shlex.quote(action.kind)} {action.size} || remove_failed=1"
            )
        lines.append("  confirmed=$((confirmed + 1))")
        lines.append('  if [[ "$remove_failed" == "1" ]]; then')
        lines.append(f"    {say} \"Confirmed, but some paths could not be removed.\"")
        lines.append("  else")
        lines.append(f"    {say} \"Removed this section.\"")
        lines.append("  fi")
        lines.append("else")
        lines.append("  skipped=$((skipped + 1))")
        lines.append(f"  {say} \"Skipped.\"")
        lines.append("fi")
        lines.append(f"{say} \"\"")

    lines.extend(
        [
            f"{say} \"----------------------------------------------------------------\"",
            f"{say} \"Finished.\"",
            f'{say} "Confirmed sections: $confirmed"',
            f'{say} "Skipped sections: $skipped"',
            "formatted=$(format_bytes \"$deleted_bytes\")",
            f'{say} "Size removed: $formatted"',
            f"{say} \"A path that appears in two sections is counted once.\"",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render(
    report: Scan,
    home: Path,
    script_path: Path | None = None,
    shell: str | None = None,
    detection: str = "",
) -> str:
    repeat_groups = repeated_groups(report.models)
    variant_rows = variant_groups(report.models)
    safe_total = sum(item.size for item in report.safe_caches)
    evernote_total = sum(item.size for item in report.evernote)
    partial_total = sum(item.size for item in report.partials)
    copy_saved = repeat_savings(repeat_groups)
    variant_saved = variant_extra_savings(variant_rows)
    review_total = (
        report.biglybt_total
        + sum(item.size for item in report.large_media)
        + sum(item.size for item in report.installers)
        + sum(item.size for item in report.other_downloads)
    )
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Disk space that can be freed",
        "",
        f"Generated {generated} for `{home}`.",
        "",
        "This report only lists files. It does not delete anything.",
        "",
    ]
    if script_path is not None:
        lines.extend(
            [
                (
                    f"The delete script is `{script_path.name}`"
                    + (f", written for {shell}" if shell else "")
                    + (f" ({detection})" if detection else "")
                    + ". It prints each section and its size, then deletes that section only if you answer `y`."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                f"Data volume: **{fmt_bytes(report.data_used)} used**, "
                f"**{fmt_bytes(report.data_free)} free**, of {fmt_bytes(report.data_total)}."
            ),
            "",
            "## Summary",
            "",
        ]
    )
    lines.append(
        md_table(
            ["Kind", "Size", "If you remove it"],
            [
                ["Safe caches", fmt_bytes(safe_total), "Apps rebuild them."],
                ["Evernote local cache", fmt_bytes(evernote_total), "Attachments download again when a note is opened."],
                ["Unfinished downloads", fmt_bytes(partial_total), "Incomplete files. The finished copies are untouched."],
                ["Repeated model files", fmt_bytes(copy_saved), "Keep the largest file of each model name."],
                [
                    "Other variants of the same base model",
                    fmt_bytes(variant_saved),
                    "Additional space if you keep only one variant of each base model, on top of the repeated-file savings.",
                ],
                ["Personal files listed below", fmt_bytes(review_total), "Your files. The sizes are listed so you can decide."],
            ],
        )
    )
    lines.extend(["", "## Safe caches", ""])
    if report.safe_caches:
        lines.append(md_table(["Path", "Size", "Why it is safe"], path_rows(report.safe_caches, home)))
        lines.append("")
        if report.cache_breakdown:
            lines.append("Largest folders inside `~/Library/Caches`:")
            lines.append("")
            lines.append(md_table(["Path", "Size"], path_rows(report.cache_breakdown, home, with_note=False)))
            lines.append("")
    else:
        lines.append("No large safe caches were found.")
        lines.append("")

    lines.extend(["## Evernote cache", ""])
    if report.evernote:
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.evernote, home)))
        lines.append("")
        lines.append("The rest of the Evernote support folder is the notebook database. This report does not list that part for deletion.")
        lines.append("")
    else:
        lines.append("No large Evernote cache was found.")
        lines.append("")

    lines.extend(["## Unfinished downloads", ""])
    if report.partials:
        lines.append(md_table(["Path", "Stored size", "Progress"], path_rows(report.partials, home)))
        lines.append("")
    else:
        lines.append("No unfinished downloads were found in Draw Things, LM Studio, or Downloads.")
        lines.append("")

    lines.extend(
        [
            "## Repeated local models",
            "",
            f"LM Studio models use **{fmt_bytes(report.models_total)}** in `~/.lmstudio/models`.",
            "",
            "A repeated model is the same name after quant and format tags are removed, stored more than once. Keeping the largest copy frees the size in this section.",
            "",
        ]
    )
    if repeat_groups:
        for key, copies in repeat_groups:
            total = sum(copy.size for copy in copies)
            largest = max(copy.size for copy in copies)
            lines.append(
                f"### `{key}` — {len(copies)} copies, {fmt_bytes(total)}, {fmt_bytes(total - largest)} recoverable"
            )
            lines.append("")
            rows = [
                [f"`{display_path(copy.path, home)}`", fmt_bytes(copy.size)]
                for copy in sorted(copies, key=lambda copy: copy.size, reverse=True)
            ]
            lines.append(md_table(["Path", "Size"], rows))
            lines.append("")
    else:
        lines.append("No repeated model names were found.")
        lines.append("")

    lines.extend(
        [
            "## Same base model, different variants",
            "",
            "These share a base model and parameter size, but the remaining name is a finetune or a text-only or uncensored build. They are not the same file.",
            "",
        ]
    )
    if variant_rows:
        for base, exacts in variant_rows:
            total = sum(copy.size for _, copies in exacts for copy in copies)
            lines.append(f"### `{base}` — {fmt_bytes(total)}")
            lines.append("")
            rows = []
            for exact, copies in exacts:
                rows.append(
                    [
                        f"`{exact}`",
                        str(len(copies)),
                        fmt_bytes(sum(copy.size for copy in copies)),
                        f"`{display_path(max(copies, key=lambda copy: copy.size).path, home)}`",
                    ]
                )
            lines.append(md_table(["Variant", "Copies", "Total", "Largest path"], rows))
            lines.append("")
    else:
        lines.append("No extra variants were found.")
        lines.append("")

    lines.extend(["## Personal files to review", ""])
    if report.biglybt_total:
        lines.append(f"### BiglyBT downloads — {fmt_bytes(report.biglybt_total)}")
        lines.append("")
        lines.append("`~/Documents/BiglyBT Downloads`")
        lines.append("")
        if report.biglybt:
            lines.append(md_table(["Path", "Size"], path_rows(report.biglybt, home, with_note=False)))
            lines.append("")
    if report.large_media:
        lines.append("### Large recordings and backup media")
        lines.append("")
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.large_media, home)))
        lines.append("")
    if report.installers:
        lines.append("### Installer downloads")
        lines.append("")
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.installers, home)))
        lines.append("")
    if report.other_downloads:
        lines.append("### Other large downloads")
        lines.append("")
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.other_downloads, home)))
        lines.append("")
    if not any([report.biglybt_total, report.large_media, report.installers, report.other_downloads]):
        lines.append("No extra personal files matched the review rules.")
        lines.append("")

    if report.other_caches:
        lines.extend(["## Other caches to check", ""])
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.other_caches, home)))
        lines.append("")

    lines.extend(["## Left as libraries", ""])
    if report.libraries:
        lines.append("These are large, and they are working libraries rather than leftover downloads.")
        lines.append("")
        lines.append(md_table(["Path", "Size", "What it is"], path_rows(report.libraries, home)))
        lines.append("")
    else:
        lines.append("No extra library folders over 1 GB were found.")
        lines.append("")

    lines.extend(["## Biggest home folders", ""])
    lines.append("Shown for context. A large folder here is not by itself a cleanup target.")
    lines.append("")
    if report.home_folders:
        lines.append(md_table(["Path", "Size"], path_rows(report.home_folders, home, with_note=False)))
        lines.append("")
    else:
        lines.append("Home folder sizes could not be measured.")
        lines.append("")

    if report.notes:
        lines.extend(["## Also", ""])
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(
        [
            "## How names are grouped",
            "",
            "Model folder names lose quant and format tags such as `Q4`, `8bit`, `MLX`, `GGUF`, `F16`, `TurboQuant`, and `M5Air`. What remains is the model name. Folders with the same remaining name are treated as copies. A shorter base key, such as `qwen3.8-27b` or `gemma-4-e4b`, groups finetunes of that base model in the variants section.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def self_check() -> None:
    expected = {
        "Qwen3.8-27B-GGUF": "qwen3.8-27b",
        "Qwen3.8-27B-Uncensored-oQ4e-mtp": "qwen3.8-27b-uncensored",
        "Qwen3.8-27B-4bit-MLX-TextOnly": "qwen3.8-27b-textonly",
        "Qwen3.6-35B-A3B-RotorQuant-MLX-3bit": "qwen3.6-35b-a3b",
        "Qwen3.6-35B-A3B-Escha-W2": "qwen3.6-35b-a3b-escha-w2",
        "Qwen3.6-27B-Miraculix-i1-GGUF": "qwen3.6-27b-miraculix",
        "gpt-oss-20b-TurboQuant-MLX-8bit": "gpt-oss-20b",
        "gpt-oss-20b-MXFP4-Q8": "gpt-oss-20b",
        "gpt-oss-20b-oQ4": "gpt-oss-20b",
        "gemma-4-E4B-it-MLX-8bit-M5Air": "gemma-4-e4b",
        "gemma-4-E4B-it-GGUF-M5Air": "gemma-4-e4b",
        "gemma-4-26B-A4B-it-QAT-MLX-4bit": "gemma-4-26b-a4b",
        "Gemma-4-MoonGem-31B-mlx-4Bit": "gemma-4-moongem-31b",
        "MiniCPM5-1B-Claude-Opus-Fable5-V2-Thinking-OptiQ-4bit": "minicpm5-1b-claude-opus-fable5-v2-thinking",
        "model-Q4_K_M": "model",
    }
    bases = {
        "qwen3.8-27b-uncensored": "qwen3.8-27b",
        "qwen3.6-35b-a3b-escha-w2": "qwen3.6-35b-a3b",
        "gemma-4-e4b": "gemma-4-e4b",
        "gemma-4-26b-a4b": "gemma-4-26b-a4b",
        "gemma-4-moongem-31b": "gemma-4-moongem-31b",
        "gpt-oss-20b": "gpt-oss-20b",
    }
    failures = []
    for name, key in expected.items():
        got = normalize_model_name(name)
        if got != key:
            failures.append(f"{name}: got {got}, want {key}")
    for exact, want in bases.items():
        got = base_key(exact)
        if got != want:
            failures.append(f"base {exact}: got {got}, want {want}")
    if failures:
        raise SystemExit("self-check failed:\n" + "\n".join(failures))
    sample_home = Path("/Users/example")
    if is_deletable(sample_home, sample_home, "dir"):
        raise SystemExit("self-check failed: home directory was treated as deletable")
    if is_deletable(sample_home / "Library", sample_home, "dir"):
        raise SystemExit("self-check failed: ~/Library was treated as deletable")
    if is_deletable(sample_home / ".lmstudio" / "models", sample_home, "dir"):
        raise SystemExit("self-check failed: the models folder was treated as deletable")
    cache = sample_home / "Library" / "Caches" / "example"
    if not is_deletable(cache, sample_home, "dir"):
        raise SystemExit("self-check failed: a cache folder was refused")
    blocked = section_if_any(
        "Blocked",
        "Must not be offered.",
        ["home"],
        [DeleteAction(sample_home, 1, "dir")],
        sample_home,
    )
    if blocked is not None:
        raise SystemExit("self-check failed: a blocked path became a section")
    sample = DeleteSection(
        "Safe caches",
        "Apps rebuild these folders.",
        ["      1.0 KB  ~/Library/Caches/example"],
        [DeleteAction(cache, 1024, "dir")],
    )
    for shell_name, shebang, prompt in (
        ("zsh", "#!/bin/zsh", 'print -n -r -- "Delete this section? [y/N] "'),
        ("bash", "#!/bin/bash", 'printf \'%s\' "Delete this section? [y/N] "'),
    ):
        script = render_delete_script([sample], sample_home, shell_name, "test")
        if not script.startswith(shebang + "\n"):
            raise SystemExit(f"self-check failed: {shell_name} shebang")
        if prompt not in script:
            raise SystemExit(f"self-check failed: {shell_name} confirmation prompt")
        if "rm -rf --" not in script or "confirm_section" not in script:
            raise SystemExit(f"self-check failed: {shell_name} delete flow")
    terminal_chain = [
        "/usr/bin/python3",
        "/bin/bash",
        "grok",
        "-zsh",
        "login",
        "/System/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal",
    ]
    shell, reason = choose_shell(terminal_chain) or ("", "")
    if shell != "zsh" or "Terminal" not in reason:
        raise SystemExit(f"self-check failed: terminal shell detection got {shell}, {reason}")
    bash_chain = ["python3", "/opt/homebrew/bin/bash", "-bash", "iTerm2"]
    shell, reason = choose_shell(bash_chain) or ("", "")
    if shell != "bash" or "iTerm2" not in reason:
        raise SystemExit(f"self-check failed: bash terminal detection got {shell}, {reason}")
    launched = choose_shell(["python3", "/bin/bash"])
    if launched != ("bash", "the shell that launched this scan (/bin/bash)"):
        raise SystemExit(f"self-check failed: launch shell detection got {launched}")
        binary = "zsh" if shell_name == "zsh" else "bash"
        check = subprocess.run([binary, "-n"], input=script, text=True, capture_output=True)
        if check.returncode != 0:
            raise SystemExit(f"self-check failed: {shell_name} syntax\n{check.stderr}")
    print("self-check ok")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan this Mac for disk space that can be freed and write a Markdown report and a delete script."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("disk-space-to-free.md"),
        help="Markdown file to write. Default: ./disk-space-to-free.md",
    )
    parser.add_argument(
        "--delete-script",
        type=Path,
        default=None,
        help="Confirmation script to write. Default: ./delete-disk-space.zsh or .bash, matching this terminal.",
    )
    parser.add_argument(
        "--shell",
        choices=("zsh", "bash"),
        default=None,
        help="Write a zsh or bash delete script instead of detecting the terminal shell.",
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=Path.home(),
        help="Home directory to scan. Default: your home directory.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Check model-name grouping and the delete-script syntax, then exit.",
    )
    return parser.parse_args(argv)


def place_output(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "darwin":
        print("This scanner uses macOS du and df. It only runs on macOS.", file=sys.stderr)
        return 2
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_check:
        self_check()
        return 0
    home = args.home.expanduser().resolve()
    if not home.is_dir():
        print(f"Home directory does not exist: {home}", file=sys.stderr)
        return 2
    report = scan(home)
    shell, detection = detect_shell(args.shell)
    log(f"terminal shell is {shell}: {detection}")
    script_path = place_output(args.delete_script or Path(default_script_name(shell)))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = build_delete_sections(report, home)
    script_text = render_delete_script(sections, home, shell, generated_at, detection)
    script_path.write_text(script_text, encoding="utf-8")
    script_path.chmod(0o755)
    log(f"wrote {script_path}")
    output = place_output(args.output)
    output.write_text(render(report, home, script_path, shell, detection), encoding="utf-8")
    log(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
