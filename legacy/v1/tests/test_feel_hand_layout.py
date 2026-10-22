import json
from pathlib import Path

from tactilelatent.feel_hand_layout import FEEL_REVISION, feel_hand_layout


def test_named_single_hand_stream_is_not_assigned_from_array_position():
    for side in ('left','right'):
        names=[f'{side}_sensor{i}' for i in range(6)]
        layout=feel_hand_layout(names)
        assert layout['channel_hands']==[side]*6
        assert layout['groups'][0]['hand']==side
        assert layout['groups'][0]['indices']==list(range(6))
        assert layout['labels']==names


def test_mixed_hand_order_is_preserved_without_reordering_data():
    names=['right_sensor2','left_sensor4','right_sensor0','left_sensor0']
    layout=feel_hand_layout(names)
    assert layout['labels']==names
    assert layout['groups'][0]['indices']==[1,3]
    assert layout['groups'][1]['indices']==[0,2]


def test_channel_count_alone_never_establishes_hand_ownership():
    names=[f'released_hand_sensor{i}' for i in range(5)]
    for environment,revision in [(None,None),('Lab','other'),('Kitchen',FEEL_REVISION)]:
        layout=feel_hand_layout(names,environment=environment,revision=revision)
        assert layout['channel_hands']==[None]*5
        assert layout['labels']==names
    verified=feel_hand_layout(names,environment='Lab',revision=FEEL_REVISION)
    assert verified['channel_hands']==['right']*5
    assert verified['labels']==[f'right_sensor{i}' for i in range(5)]


def test_all_acquired_lab_episodes_have_explicit_right_hand_ownership():
    path=Path('processing/data_queue_v2/feel/manifest.json')
    if not path.exists():return
    manifest=json.loads(path.read_text())
    assert manifest['revision']==FEEL_REVISION
    lab=[r for r in manifest['episodes'] if r['environment']=='Lab']
    assert len(lab)==19
    for row in lab:
        layout=feel_hand_layout(row['targets']['channel_order'],environment=row['environment'],
            revision=manifest['revision'])
        assert layout['channel_hands']==['right']*5
        assert layout['measured_hands']==['right'] and layout['unmeasured_hands']==['left']
    assert next(r for r in lab if r['episode_id']=='Lab_P11')['split']=='val'
