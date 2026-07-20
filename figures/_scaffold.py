"""Helper utilities for figure-generation scripts.

Each fig*.py reads a small set of paper/benchmark_results/B*.json files
declaratively and produces a paper/figures/<name>.pdf. Until Wave 2
populates them, the scripts' main() raise NotImplementedError so that
they fail loudly when run, ensuring we never silently use stale figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RESULTS = REPO_ROOT / "paper" / "benchmark_results"
FIGURES_DIR = REPO_ROOT / "paper" / "figures"


def load_benchmark(benchmark_id: str) -> dict:
    path = BENCHMARK_RESULTS / f"{benchmark_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Required benchmark output not yet produced: {path}\n"
            f"Run the corresponding benchmark script before invoking this figure script."
        )
    return json.loads(path.read_text())


def load_benchmark_or_synth(benchmark_id: str) -> tuple[dict, bool]:
    """Try to load the real benchmark JSON; fall back to synthetic.

    Returns (data, is_real). When is_real is False, the figure script should
    stamp a 'PRELIMINARY (synthetic placeholder)' watermark via
    ``_palette.watermark_preliminary()``.
    """
    path = BENCHMARK_RESULTS / f"{benchmark_id}.json"
    if path.exists():
        return json.loads(path.read_text()), True
    from _synth import synthesize
    return synthesize(benchmark_id), False


def scaffold_main(figure_name: str, requires: list[str]):
    """Print a clear 'not yet implemented' message and exit non-zero."""
    print(f"Figure scaffold '{figure_name}' is not yet implemented.", file=sys.stderr)
    if requires:
        print(f"This figure will read: {', '.join(requires)}", file=sys.stderr)
    else:
        print("This figure has no benchmark JSON dependency (diagram / screenshots).", file=sys.stderr)
    sys.exit(2)


def placeholder_main(
    figure_id: str,
    figure_title: str,
    prompt_filename: str,
    output_filenames: list[str] | None = None,
) -> int:
    """Generate a clean placeholder PDF for a figure not yet finalised.

    The placeholder shows the figure id, title, a pointer to the Cowork prompt
    markdown, and a PLACEHOLDER watermark. Use this while real figures are
    being produced externally in Cowork.

    Parameters
    ----------
    figure_id : str
        Short id shown in the placeholder, e.g. "Fig. 1", "ED Fig. 3".
    figure_title : str
        One-line descriptive title, e.g. "Pipeline architecture & sync loop".
    prompt_filename : str
        Filename of the Cowork prompt under paper/figures/prompts/, e.g.
        "fig1_pipeline.md". The placeholder shows the full repo-relative path.
    output_filenames : list[str] | None
        PDF basenames (without directory) to write under paper/figures/.
        Defaults to a single entry derived from this script's filename stem.

    Returns
    -------
    int : exit code (0 on success).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from pathlib import Path
    import sys as _sys

    # Lazy import so this helper has no top-level side effect
    sys_path_dir = str(Path(__file__).resolve().parent)
    if sys_path_dir not in _sys.path:
        _sys.path.insert(0, sys_path_dir)
    from _palette import apply_rcparams  # noqa: E402

    if output_filenames is None:
        # Derive default from caller's script name
        import inspect
        caller_file = Path(inspect.stack()[1].filename)
        output_filenames = [caller_file.stem + ".pdf"]

    apply_rcparams()

    fig_w_in = 6.5
    fig_h_in = 3.5
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")

    # Outer border
    ax.add_patch(Rectangle(
        (2, 2), 96, 96,
        facecolor="#FAFAFA", edgecolor="#666666", linewidth=1.0,
    ))

    # Figure id (top-left, large)
    ax.text(
        6, 88, figure_id,
        fontsize=16, fontweight="bold", color="#0072B2",
    )

    # Title under id
    ax.text(
        6, 78, figure_title,
        fontsize=11, color="#222222",
    )

    # Body: pointer to Cowork prompt
    prompt_path_rel = f"paper/figures/prompts/{prompt_filename}"
    ax.text(
        50, 55, "Placeholder — figure pending external generation.",
        ha="center", va="center", fontsize=10, color="#444444",
    )
    ax.text(
        50, 47, "Full prompt for Cowork is at:",
        ha="center", va="center", fontsize=9, color="#444444",
    )
    ax.text(
        50, 40, prompt_path_rel,
        ha="center", va="center", fontsize=10, family="monospace",
        color="#0072B2", weight="bold",
    )

    # Diagonal PLACEHOLDER watermark
    ax.text(
        50, 50, "PLACEHOLDER",
        ha="center", va="center",
        rotation=30, fontsize=44, color="#CC79A7", alpha=0.15,
        weight="bold", zorder=100,
    )

    # Footer hint
    ax.text(
        50, 8,
        "Once the final figure exists, replace this PDF in paper/figures/ "
        "and the paper will pick it up automatically.",
        ha="center", va="center", fontsize=7, color="#888888", style="italic",
    )

    output_paths = []
    for name in output_filenames:
        out = FIGURES_DIR / name
        plt.savefig(out)
        output_paths.append(str(out))
    plt.close(fig)
    for p in output_paths:
        print(f"Wrote {p}")
    return 0
