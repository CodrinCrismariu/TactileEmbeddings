"""Select a bounded, category-diverse original EgoTouch subset and convert it."""
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from .prepare_pilot import convert
from .train import check_storage


def main():
    repo = "zhouzhoujy/EgoTouch"
    revision = "cfdbb0ac31cc2af4247943820aa250575e7e6637"
    raw_root = Path("datasets/zhouzhoujy__EgoTouch_expanded")
    output = Path("processing/egotouch_expanded")
    raw_splits = json.loads(Path("datasets/zhenyuxie-zhzh__EgoTouch_hdf5/split.json").read_text())
    files = {s.rfilename: s.size for s in HfApi().dataset_info(repo, revision=revision, files_metadata=True).siblings}
    manifest = {"repo": repo, "revision": revision, "episodes": []}
    selected = {"train": [], "val": []}
    budget = 500_000_000
    for split, per_category in (("train", 2), ("val", 1)):
        counts = {}
        tasks = set()
        for entry in raw_splits[split]:
            relative = Path(*Path(entry).parts[-3:])
            category, task = relative.parts[:2]
            if counts.get(category, 0) >= per_category or (category, task) in tasks:
                continue
            prefix = relative.with_suffix("").as_posix()
            wanted = [prefix + "/" + key for key in ("chest.mp4", "pressure_grids.npz", "jq_pressure.json")]
            if not all(key in files for key in wanted):
                continue
            size = sum(files[k] for k in wanted)
            if size > 40_000_000 or size > budget:
                continue
            check_storage(reserve=size + 1_000_000_000)
            for key in wanted:
                hf_hub_download(repo, key, repo_type="dataset", revision=revision, local_dir=raw_root)
            try:
                convert(raw_root / prefix, output / relative)
            except ValueError as e:
                print(f"REJECTED {prefix}: {e}", flush=True)
                budget -= size
                continue
            budget -= size
            selected[split].append(relative.as_posix())
            counts[category] = counts.get(category, 0) + 1
            tasks.add((category, task))
            manifest["episodes"].append({"split": split, "episode": prefix, "source_bytes": size})
            output.mkdir(parents=True, exist_ok=True)
            (output / "split.json").write_text(json.dumps(selected, indent=2))
            (output / "provenance.json").write_text(json.dumps(manifest, indent=2))
            print(split, prefix, size, flush=True)
    print({k: len(v) for k, v in selected.items()}, flush=True)


if __name__ == "__main__":
    main()
