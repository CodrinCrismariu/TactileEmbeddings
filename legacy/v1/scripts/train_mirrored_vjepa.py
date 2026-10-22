#!/usr/bin/env python3
"""Opt-in bilateral mirroring; ordinary checkpoints need no inference changes."""
from tactilelatent.expansion_runtime import configure
configure()


def install(trainer):
    from tactilelatent.mirror_joint import bind,read_preparation
    original=trainer.prepare
    def prepare(cfg,allow_pilot=False):
        # Fail on unverified requested mappings before initializing a backbone
        # or trying unrelated expansion prerequisites.
        read_preparation(cfg)
        if cfg.get('data_expansion'):
            from tactilelatent.expansion_joint import prepare as expanded_prepare
            result=expanded_prepare(cfg,original,trainer,allow_pilot)
        else:result=original(cfg,allow_pilot)
        return bind(cfg,result)
    trainer.prepare=prepare
    return trainer


def main():
    import json,sys
    from pathlib import Path
    if not any(x=='--config' or x.startswith('--config=') for x in sys.argv[1:]):
        sys.argv.extend(['--config','configs/joint_vjepa21_spatial_current_32d_mirror.json'])
    from tactilelatent import train_vjepa_markers as trainer
    args=sys.argv[1:]
    path=next((v.split('=',1)[1] for v in args if v.startswith('--config=')),None)
    if path is None:path=args[args.index('--config')+1]
    cfg=json.loads(Path(path).read_text())
    if cfg.get('data_expansion'):
        from tactilelatent.expansion_joint import ExpandedTeachers
        trainer.MixedMarkerTeachers=ExpandedTeachers
        trainer.MARKER_SOURCES=(*trainer.MARKER_SOURCES,'baihu')
    install(trainer).main()


if __name__=='__main__':main()
