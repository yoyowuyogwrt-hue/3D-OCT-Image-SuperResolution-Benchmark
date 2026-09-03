"""Patch a BATDiff checkout so its reference image can come from DIP instead of bicubic.

BATDiff builds every one of its training targets from a single reference image
`x_ref`. In the released code `x_ref` is the low-resolution input stretched back
up with bicubic interpolation, which carries no real high-frequency content. The
a trous wavelet decomposition is then computed on that reference, so each scale
the diffusion model learns is derived from interpolated detail.

The contribution of this project is to substitute a Deep Image Prior
reconstruction for that bicubic reference. Because the substitution happens
before the wavelet decomposition, it changes what the diffusion model is trained
to reproduce, rather than post-processing whatever it produces.

Usage:
    python scripts/batdiff_dip_patch.py --batdiff-root external/BATDiff
    python scripts/batdiff_dip_patch.py --batdiff-root external/BATDiff --check

The script is idempotent: running it twice leaves the checkout unchanged.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Edit:
    """One exact-string substitution inside one file of the BATDiff checkout."""

    relative_path: str
    label: str
    old: str
    new: str

    def is_applied(self, text: str) -> bool:
        return self.new in text

    def can_apply(self, text: str) -> bool:
        return text.count(self.old) == 1


# --------------------------------------------------------------------------
# Contribution: route the wavelet reference through an external image.
# --------------------------------------------------------------------------

CONTRIBUTION_EDITS: tuple[Edit, ...] = (
    Edit(
        relative_path="BATDiff/functions.py",
        label="create_img_scales() accepts xref_path",
        old=(
            "                      hf_boost=1.0,\n"
            "                      keep_same_as_input=True):\n"
        ),
        new=(
            "                      hf_boost=1.0,\n"
            "                      keep_same_as_input=True,\n"
            "                      xref_path=None):\n"
        ),
    ),
    Edit(
        relative_path="BATDiff/functions.py",
        label="wavelet reference is DIP output when xref_path is given",
        old=(
            "    x_ref_pil = lr_image.resize(image_size, Image.BICUBIC)\n"
            "    x_ref = np.asarray(x_ref_pil).astype(np.float32) / 255.0\n"
        ),
        new=(
            "    if xref_path is not None:\n"
            "        # Take the reference from an external reconstruction (a Deep Image\n"
            "        # Prior output) instead of interpolating the LR input. The a trous\n"
            "        # decomposition below runs on x_ref, so every per-scale training\n"
            "        # target the diffusion model sees changes with it.\n"
            "        x_ref_pil = Image.open(xref_path).convert(\"RGB\")\n"
            "        if x_ref_pil.size != tuple(image_size):\n"
            "            x_ref_pil = x_ref_pil.resize(image_size, Image.BICUBIC)\n"
            "    else:\n"
            "        x_ref_pil = lr_image.resize(image_size, Image.BICUBIC)\n"
            "    x_ref = np.asarray(x_ref_pil).astype(np.float32) / 255.0\n"
        ),
    ),
    Edit(
        relative_path="main.py",
        label="--xref_image command line flag",
        old=(
            '    parser.add_argument("--sr_factor", help="super-resolution upscaling factor",'
            " default=8, type=int)\n"
        ),
        new=(
            '    parser.add_argument("--sr_factor", help="super-resolution upscaling factor",'
            " default=8, type=int)\n"
            '    parser.add_argument("--xref_image", help=\'reference image used instead of'
            " bicubic upsampling, e.g. a DIP reconstruction.', default=None, type=str)\n"
        ),
    ),
    Edit(
        relative_path="main.py",
        label="pass --xref_image into create_img_scales",
        old="    keep_same_as_input=False\n)\n",
        new="    keep_same_as_input=False,\n    xref_path=args.xref_image\n)\n",
    ),
    Edit(
        relative_path="main.py",
        label="sample mode honours --xref_image",
        old="        xref_img = lr_img.resize(hr_target_size, Image.BICUBIC)\n",
        new=(
            "        if args.xref_image is not None:\n"
            "            xref_img = Image.open(args.xref_image).convert(\"RGB\")\n"
            "            if xref_img.size != tuple(hr_target_size):\n"
            "                xref_img = xref_img.resize(hr_target_size, Image.BICUBIC)\n"
            "        else:\n"
            "            xref_img = lr_img.resize(hr_target_size, Image.BICUBIC)\n"
        ),
    ),
)


# --------------------------------------------------------------------------
# Compatibility: unrelated to the contribution, needed to run on current Colab.
# --------------------------------------------------------------------------

COMPATIBILITY_EDITS: tuple[Edit, ...] = (
    Edit(
        relative_path="clip/clip.py",
        label="pkg_resources fallback (removed in setuptools >= 81)",
        old="from pkg_resources import packaging\n",
        new=(
            "try:\n"
            "    from pkg_resources import packaging\n"
            "except ImportError:  # setuptools >= 81 no longer ships pkg_resources\n"
            "    import packaging\n"
            "    import packaging.version\n"
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batdiff-root",
        type=Path,
        default=PROJECT_ROOT / "external" / "BATDiff",
        help="Path to the cloned BATDiff repository.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report patch status without writing anything.",
    )
    return parser.parse_args()


def apply_edits(root: Path, edits: tuple[Edit, ...], check_only: bool) -> tuple[int, int]:
    """Apply `edits` under `root`. Returns (n_applied_now, n_already_applied)."""
    applied = 0
    already = 0

    by_file: dict[str, list[Edit]] = {}
    for edit in edits:
        by_file.setdefault(edit.relative_path, []).append(edit)

    for relative_path, file_edits in by_file.items():
        path = root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

        text = path.read_text(encoding="utf-8")
        original = text

        for edit in file_edits:
            if edit.is_applied(text):
                print(f"  [already] {relative_path}: {edit.label}")
                already += 1
                continue
            if not edit.can_apply(text):
                occurrences = text.count(edit.old)
                raise RuntimeError(
                    f"Cannot patch {relative_path} ({edit.label}): expected exactly one "
                    f"match for the target snippet, found {occurrences}. The upstream "
                    f"file has probably changed."
                )
            text = text.replace(edit.old, edit.new, 1)
            print(f"  [{'would patch' if check_only else 'patched'}] "
                  f"{relative_path}: {edit.label}")
            applied += 1

        if text != original and not check_only:
            path.write_text(text, encoding="utf-8")

    return applied, already


def main() -> int:
    args = parse_args()
    root = args.batdiff_root.resolve()

    if not (root / "main.py").exists():
        print(f"error: {root} does not look like a BATDiff checkout.", file=sys.stderr)
        print("Clone it first:", file=sys.stderr)
        print("  git clone https://github.com/MaryamHeidari-1994/BATDiff external/BATDiff",
              file=sys.stderr)
        return 1

    print(f"BATDiff checkout: {root}")
    print("\nContribution: DIP reference in place of bicubic")
    contribution_new, contribution_old = apply_edits(root, CONTRIBUTION_EDITS, args.check)

    print("\nCompatibility fixes")
    compat_new, compat_old = apply_edits(root, COMPATIBILITY_EDITS, args.check)

    total_new = contribution_new + compat_new
    total_old = contribution_old + compat_old

    print()
    if args.check:
        print(f"{total_old} edit(s) already present, {total_new} outstanding.")
        return 1 if total_new else 0

    print(f"{total_new} edit(s) applied, {total_old} already present.")
    print("\nBaseline (unchanged behaviour) omits --xref_image.")
    print("The DIP-conditioned run adds:  --xref_image /path/to/dip.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
