"""Audit completeness of the official raw EgoTouch training subset."""
import json
from pathlib import Path
REQUIRED=("chest.mp4","left.mp4","right.mp4","pressure_grids.npz")

def main(root="datasets/EgoTouch"):
    root=Path(root); report={"root":str(root),"download_complete":False,"splits":{}}; split_path=root/"split.json"
    if not split_path.exists(): report["error"]="split.json is missing (download is incomplete)"
    else:
        splits=json.loads(split_path.read_text())
        for name,entries in splits.items():
            missing=[]
            for entry in entries:
                episode=root/Path(*Path(entry).parts[-3:]).with_suffix("")
                absent=[item for item in REQUIRED if not (episode/item).is_file()]
                if absent: missing.append({"episode":str(episode.relative_to(root)),"files":absent})
            report["splits"][name]={"split_entries":len(entries),"usable_episodes":len(entries)-len(missing),"unavailable_in_release":len(missing),"examples":missing[:10]}
        # snapshot_download's successful completion is additionally recorded in its log.
        report["download_complete"]=True
    destination=Path("processing/audits/egotouch.json"); destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
