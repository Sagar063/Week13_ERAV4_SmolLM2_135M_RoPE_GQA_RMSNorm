"""
update_readme.py

Updates README.md by auto-inserting:
1) out_smollm2_scratch/train.log into the TRAIN_LOG block (collapsible <details>)
2) ablation_results_500steps/ablations.csv into the ABLATION_TABLE block as a Markdown table
3) proof_sdpa_on/profiles/sdpa_on/sdpa_on_top_ops.txt into SDPA_ON_TOPS block
4) proof_sdpa_off/profiles/sdpa_off/sdpa_off_top_ops.txt into SDPA_OFF_TOPS block

Usage:
  python update_readme.py
"""

from __future__ import annotations

import os
import re
import pandas as pd


README_PATH = "README.md"
TRAIN_LOG_PATH = os.path.join("out_smollm2_scratch", "train.log")
ABLATION_CSV_PATH = os.path.join("ablation_results_500steps", "ablations.csv")
SDPA_ON_TOPS_PATH = os.path.join("proof_sdpa_on", "profiles", "sdpa_on", "sdpa_on_top_ops.txt")
SDPA_OFF_TOPS_PATH = os.path.join("proof_sdpa_off", "profiles", "sdpa_off", "sdpa_off_top_ops.txt")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def make_details_block(log_text: str, title: str) -> str:
    # Keep it safe for Markdown rendering (normalize newlines)
    log_text = log_text.replace("\r\n", "\n").strip()
    return (
        "<details>\n"
        f"  <summary><b>Click to expand: {title}</b></summary>\n\n"
        "```text\n"
        f"{log_text}\n"
        "```\n\n"
        "</details>"
    )


def df_to_markdown_table(df: pd.DataFrame) -> str:
    # Limit float precision for readability
    df2 = df.copy()
    for col in df2.columns:
        if pd.api.types.is_float_dtype(df2[col]):
            df2[col] = df2[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else x)
    return df2.to_markdown(index=False)


def replace_block(md: str, begin_marker: str, end_marker: str, new_body: str) -> str:
    pattern = re.compile(
        rf"({re.escape(begin_marker)})(.*?)({re.escape(end_marker)})",
        re.DOTALL,
    )
    m = pattern.search(md)
    if not m:
        raise RuntimeError(f"Could not find markers: {begin_marker} ... {end_marker}")
    # Replace the content between markers, keeping the markers themselves
    return md[: m.start(2)] + new_body + md[m.end(2) :]


def main():
    md = read_text(README_PATH)

    # 1) Train log insertion
    if os.path.exists(TRAIN_LOG_PATH):
        log_text = read_text(TRAIN_LOG_PATH)
        details = make_details_block(log_text, "out_smollm2_scratch/train.log")
    else:
        details = (
            "<details>\n"
            "  <summary><b>Click to expand: out_smollm2_scratch/train.log</b></summary>\n\n"
            "  Train log not found. Run training to create it.\n\n"
            "</details>"
        )
    md = replace_block(md, "<!-- TRAIN_LOG_BEGIN -->", "<!-- TRAIN_LOG_END -->", details)

    # 2) Ablation CSV insertion
    if os.path.exists(ABLATION_CSV_PATH):
        df = pd.read_csv(ABLATION_CSV_PATH)
        table = df_to_markdown_table(df)
        ablation_md = table
    else:
        ablation_md = f"Ablation CSV not found at `{ABLATION_CSV_PATH}`. Run `train_ablation.py --run_all` first."
    md = replace_block(md, "<!-- ABLATION_TABLE_BEGIN -->", "<!-- ABLATION_TABLE_END -->", ablation_md)

    # 3) SDPA ON top ops insertion
    if os.path.exists(SDPA_ON_TOPS_PATH):
        tops_text = read_text(SDPA_ON_TOPS_PATH)
        sdpa_on_details = make_details_block(tops_text, "sdpa_on top CUDA operations")
    else:
        sdpa_on_details = (
            "<details>\n"
            "  <summary><b>Click to expand: sdpa_on top CUDA operations</b></summary>\n\n"
            f"  Profile file not found at `{SDPA_ON_TOPS_PATH}`. Run profiling for sdpa_on variant first.\n\n"
            "</details>"
        )
    md = replace_block(md, "<!-- SDPA_ON_TOPS_BEGIN -->", "<!-- SDPA_ON_TOPS_END -->", sdpa_on_details)

    # 4) SDPA OFF top ops insertion
    if os.path.exists(SDPA_OFF_TOPS_PATH):
        tops_text = read_text(SDPA_OFF_TOPS_PATH)
        sdpa_off_details = make_details_block(tops_text, "sdpa_off top CUDA operations")
    else:
        sdpa_off_details = (
            "<details>\n"
            "  <summary><b>Click to expand: sdpa_off top CUDA operations</b></summary>\n\n"
            f"  Profile file not found at `{SDPA_OFF_TOPS_PATH}`. Run profiling for sdpa_off variant first.\n\n"
            "</details>"
        )
    md = replace_block(md, "<!-- SDPA_OFF_TOPS_BEGIN -->", "<!-- SDPA_OFF_TOPS_END -->", sdpa_off_details)

    write_text(README_PATH, md)
    print("README.md updated successfully.")


if __name__ == "__main__":
    main()
