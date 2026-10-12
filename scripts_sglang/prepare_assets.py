"""Download pinned BrowseComp-Plus data, Qwen retrieval assets, and the judge."""

import argparse
import csv
import json
import os
from pathlib import Path


REVISIONS = {
    "Qwen/Qwen3-Embedding-8B": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
    "Qwen/Qwen3-32B": "9216db5781bf21249d130ec9da846c4624c16137",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Tevatron/browsecomp-plus-indexes": "b3f37f70c33829eb09d04784a54277a31871fd63",
    "Tevatron/browsecomp-plus-corpus": "b27b02bc3e45511b8b82a13e6f90ce761df726f6",
    "Tevatron/browsecomp-plus": "144cff8e35b5eaef7e526346aa60774a9deb941f",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / ".cache/huggingface"))
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    import sys

    sys.path.insert(0, str(root))
    from scripts_build_index.decrypt_dataset import DEFAULT_CANARY, transform_decrypt

    for name in ("data", "topics-qrels", "indexes", "logs"):
        (root / name).mkdir(exist_ok=True)
    manifest_path = root / "scripts_sglang/assets.local.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["revisions"] = REVISIONS

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    def model(key, model_id, *, tokenizer_only=False):
        print(f"Preparing {model_id}", flush=True)
        patterns = ["*.json", "*.jinja", "*.txt", "*.model"]
        if not tokenizer_only:
            patterns.append("*.safetensors")
        manifest[key] = {
            "id": model_id,
            "revision": REVISIONS[model_id],
            "path": snapshot_download(
                model_id,
                revision=REVISIONS[model_id],
                allow_patterns=patterns,
                max_workers=4,
            ),
        }
        save()

    model("embedding", "Qwen/Qwen3-Embedding-8B")
    model("snippet_tokenizer", "Qwen/Qwen3-0.6B", tokenizer_only=True)
    index_id = "Tevatron/browsecomp-plus-indexes"
    snapshot_download(
        index_id,
        repo_type="dataset",
        revision=REVISIONS[index_id],
        allow_patterns=["qwen3-embedding-8b/*"],
        local_dir=root / "indexes",
        max_workers=4,
    )
    manifest["index_glob"] = str(root / "indexes/qwen3-embedding-8b/corpus.shard*.pkl")
    save()

    corpus_path = root / "data/corpus"
    if not (corpus_path / "dataset_info.json").exists():
        corpus_id = "Tevatron/browsecomp-plus-corpus"
        corpus = load_dataset(corpus_id, revision=REVISIONS[corpus_id], split="train")
        corpus.save_to_disk(corpus_path)
        print(f"Saved {len(corpus)} corpus documents", flush=True)
    manifest["corpus_path"] = str(corpus_path)
    save()

    dataset_id = "Tevatron/browsecomp-plus"
    dataset = load_dataset(dataset_id, revision=REVISIONS[dataset_id], split="test")
    with (root / "data/browsecomp_plus_decrypted.jsonl").open("w") as output:
        with (root / "topics-qrels/queries.tsv").open("w", newline="") as queries:
            writer = csv.writer(queries, delimiter="\t")
            for record in dataset:
                row = transform_decrypt(record, DEFAULT_CANARY, {"query_id"})
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                query = str(row["query"]).replace("\t", " ").replace("\n", " ")
                writer.writerow([str(row["query_id"]), query])
    manifest["query_count"] = len(dataset)
    save()
    print(
        f"Prepared {len(dataset)} benchmark queries (contents kept local)", flush=True
    )
    if not args.skip_judge:
        model("judge", "Qwen/Qwen3-32B")
    print(f"Asset manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
