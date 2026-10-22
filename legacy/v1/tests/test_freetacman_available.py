import importlib.util
import json
from pathlib import Path

import pytest

from tactilelatent.aligned_frames import json_hash
from tactilelatent.marker_data import episode_key


def test_available_snapshot_is_fixed_and_keeps_episode_splits(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('available_runner',Path(__file__).parents[1]/'scripts/train_freetacman_available_markers.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=[dict(source='freetacman',episode_id=name,split=split,frames=20) for name,split in [('a','train'),('b','val'),('c','train')]]
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(episodes=rows)))
    root=tmp_path/'targets';output=tmp_path/'run';output.mkdir()
    def complete(row):
        folder=root/row['source']/row['split']/episode_key(row);folder.mkdir(parents=True);(folder/'complete.json').write_text('{}')
    complete(rows[0]);complete(rows[1])
    monkeypatch.setattr(runner,'manifest_signature',lambda _:dict(original=True))
    def snapshot(path,targets,source):
        selected=json.loads(Path(path).read_text())['episodes']
        return dict(source_snapshot_sha256=json_hash(selected),episodes=selected)
    monkeypatch.setattr(runner,'snapshot',snapshot)
    pinned=runner.freeze_available(str(manifest),str(root),output)
    assert pinned['selection']['selected_by_split']=={'train':1,'val':1}
    assert pinned['selection']['missing_episode_ids']==['c']
    complete(rows[2])
    resumed=runner.freeze_available(str(manifest),str(root),output)
    assert resumed==pinned and len(resumed['episodes'])==2
    monkeypatch.setattr(runner,'manifest_signature',lambda _:dict(original=False))
    with pytest.raises(ValueError,match='snapshot changed'):runner.freeze_available(str(manifest),str(root),output)
