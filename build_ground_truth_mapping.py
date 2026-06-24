#!/usr/bin/env python3
"""Build and validate the hand-selected ground-truth image mapping.

Each numeric image name is a 1-based index into its page's ``<page>.txt``
file.  The value on that line is in turn a 1-based index into
``telugu_vocab.txt``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Mapping:
    image: Path
    vocab_id: int
    label: str


class MappingError(ValueError):
    """Raised when the ground-truth data is inconsistent."""


def natural_key(path: Path) -> tuple[tuple[int, object], ...]:
    """Sort path components numerically where possible."""
    key: list[tuple[int, object]] = []
    for part in path.parts:
        for token in re.split(r"(\d+)", part):
            if token:
                key.append((0, int(token)) if token.isdigit() else (1, token))
        key.append((2, ""))
    return tuple(key)


def read_lines(path: Path) -> list[str]:
    try:
        # utf-8-sig also accepts regular UTF-8 while removing an accidental BOM.
        return path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise MappingError(f"could not read {path}: {exc}") from exc


def read_vocab(path: Path) -> list[str]:
    vocab = read_lines(path)
    if not vocab:
        raise MappingError(f"vocabulary is empty: {path}")
    for line_number, label in enumerate(vocab, start=1):
        if not label:
            raise MappingError(f"empty vocabulary entry at {path}:{line_number}")
    return vocab


def page_directories(sample_root: Path) -> list[Path]:
    if not sample_root.is_dir():
        raise MappingError(f"sample root is not a directory: {sample_root}")

    pages = [
        path
        for path in sample_root.glob("*/*/*")
        if path.is_dir()
    ]
    if not pages:
        raise MappingError(
            f"no split/writer/page directories found below {sample_root}"
        )
    return sorted(pages, key=lambda path: natural_key(path.relative_to(sample_root)))


def build_mappings(
    sample_root: Path,
    vocab: list[str],
    manifests: dict[str, dict[str, str]] | None,
) -> list[Mapping]:
    mappings: list[Mapping] = []
    vocab_ids: dict[str, list[int]] = {}
    for vocab_id, label in enumerate(vocab, start=1):
        vocab_ids.setdefault(label, []).append(vocab_id)

    for page_dir in page_directories(sample_root):
        page_name = page_dir.name
        labels_path = page_dir / f"{page_name}.txt"
        if not labels_path.is_file():
            raise MappingError(f"missing page label file: {labels_path}")

        raw_ids = read_lines(labels_path)
        images = sorted(page_dir.glob("*.jpg"), key=lambda path: natural_key(path))
        if len(images) != len(raw_ids):
            raise MappingError(
                f"{page_dir}: {len(images)} JPGs but {len(raw_ids)} label lines"
            )

        numeric_images: dict[int, Path] = {}
        unusual_images: list[Path] = []
        for image in images:
            if image.stem.isdigit() and int(image.stem) >= 1:
                image_number = int(image.stem)
                if image_number in numeric_images:
                    raise MappingError(
                        f"duplicate numeric image name in {page_dir}: "
                        f"{numeric_images[image_number].name}, {image.name}"
                    )
                numeric_images[image_number] = image
            else:
                unusual_images.append(image)

        extra = sorted(set(numeric_images) - set(range(1, len(raw_ids) + 1)))
        if extra:
            raise MappingError(f"{page_dir}: JPG numbers without label lines {extra}")
        if not unusual_images:
            missing = sorted(set(range(1, len(raw_ids) + 1)) - set(numeric_images))
            if missing:
                raise MappingError(f"{page_dir}: missing JPG numbers {missing}")

        for image_number, image in sorted(numeric_images.items()):
            raw_id = raw_ids[image_number - 1]
            value = raw_id.strip()
            if not value.isdigit() or int(value) < 1:
                raise MappingError(
                    f"invalid vocab ID {raw_id!r} at {labels_path}:{image_number}"
                )
            vocab_id = int(value)
            if vocab_id > len(vocab):
                raise MappingError(
                    f"vocab ID {vocab_id} at {labels_path}:{image_number} "
                    f"exceeds the {len(vocab)}-entry vocabulary"
                )
            mappings.append(Mapping(image, vocab_id, vocab[vocab_id - 1]))

        for image in unusual_images:
            if manifests is None:
                raise MappingError(
                    f"cannot map non-numeric JPG without manifest checking: {image}"
                )
            relative = image.relative_to(sample_root)
            split = relative.parts[0]
            canonical_path = (Path("TeluguSeg") / relative).as_posix()
            label = manifests[split].get(canonical_path)
            if label is None:
                raise MappingError(f"image absent from {split}.txt: {canonical_path}")
            matching_ids = vocab_ids.get(label, [])
            if len(matching_ids) != 1:
                raise MappingError(
                    f"cannot uniquely map {canonical_path} label {label!r} to vocab; "
                    f"found IDs {matching_ids}"
                )
            print(
                f"warning: used {split}.txt for non-numeric image {relative.as_posix()}",
                file=sys.stderr,
            )
            mappings.append(Mapping(image, matching_ids[0], label))

    return sorted(mappings, key=lambda item: natural_key(item.image.relative_to(sample_root)))


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    if not path.is_file():
        raise MappingError(f"canonical manifest not found: {path}")
    for line_number, line in enumerate(read_lines(path), start=1):
        try:
            image, label = line.split(maxsplit=1)
        except ValueError as exc:
            raise MappingError(f"malformed manifest line at {path}:{line_number}") from exc
        if image in entries:
            raise MappingError(f"duplicate image in manifest {path}: {image}")
        entries[image] = label
    return entries


def load_manifests(sample_root: Path, project_root: Path) -> dict[str, dict[str, str]]:
    splits = {page.relative_to(sample_root).parts[0] for page in page_directories(sample_root)}
    return {split: read_manifest(project_root / f"{split}.txt") for split in splits}


def validate_manifests(
    mappings: list[Mapping],
    sample_root: Path,
    manifests: dict[str, dict[str, str]],
) -> None:

    for mapping in mappings:
        relative = mapping.image.relative_to(sample_root)
        split = relative.parts[0]
        canonical_path = (Path("TeluguSeg") / relative).as_posix()
        canonical_label = manifests[split].get(canonical_path)
        if canonical_label is None:
            raise MappingError(f"image absent from {split}.txt: {canonical_path}")
        if canonical_label != mapping.label:
            raise MappingError(
                f"label mismatch for {canonical_path}: vocab gives "
                f"{mapping.label!r}, {split}.txt gives {canonical_label!r}"
            )


def write_mapping(
    output: Path, mappings: list[Mapping], project_root: Path
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(("image_path", "vocab_id", "label"))
            for mapping in mappings:
                try:
                    display_path = mapping.image.relative_to(project_root)
                except ValueError:
                    display_path = mapping.image
                writer.writerow((display_path.as_posix(), mapping.vocab_id, mapping.label))
        temporary.replace(output)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise MappingError(f"could not write {output}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-root",
        type=Path,
        default=project_root / "ground_truth" / "sample_set",
        help="root containing split/writer/page directories",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=project_root / "telugu_vocab.txt",
        help="one-label-per-line vocabulary file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "ground_truth" / "mapping" / "selected_ground_truth_pages.txt",
        help="output TSV mapping file",
    )
    parser.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help=(
            "do not compare labels with <split>.txt manifests "
            "(requires every JPG name to be numeric)"
        ),
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sample_root = args.sample_root.resolve()
        vocab = read_vocab(args.vocab.resolve())
        manifests = None
        if not args.skip_manifest_check:
            manifests = load_manifests(sample_root, args.project_root)
        mappings = build_mappings(sample_root, vocab, manifests)
        if manifests is not None:
            validate_manifests(mappings, sample_root, manifests)
        write_mapping(args.output.resolve(), mappings, args.project_root)
    except MappingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {len(mappings)} mappings from "
        f"{len({mapping.image.parent for mapping in mappings})} pages to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
