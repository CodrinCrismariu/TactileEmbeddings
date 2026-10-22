"""Verified FEEL hand ownership, independent of a channel's array position.

The pinned Lab release uses unnamed force_aligned_.csv files. Its official
preprocess.py applies right-hand ignore rules; the project labels Lab previews
P<n>_right_hand_*. Kitchen's force_left/right filenames identify each hand.
Native channel IDs and stored values remain immutable for provenance/caches.
"""
import re


FEEL_REVISION = '0f8179649b50afcfd684c87784c67787871db361'
POLICY = 'feel-hand-ownership-v1'
EVIDENCE = 'https://www.cs.umd.edu/~edessale/feel'


def feel_hand_layout(channels, *, environment=None, revision=None):
    channels = list(channels)
    labels = list(channels)
    hands = [None] * len(channels)
    lab = (environment == 'Lab' and revision == FEEL_REVISION
           and channels == [f'released_hand_sensor{i}' for i in range(5)])
    if lab:
        labels = [f'right_sensor{i}' for i in range(5)]
        hands = ['right'] * 5
        basis = 'Pinned Lab preprocessing uses right-hand rules; official Lab previews are labelled right_hand.'
    else:
        # An explicitly right-labelled single-hand stream stays right even when
        # its channels start at index 0. No first-group-is-left assumption.
        for i, channel in enumerate(channels):
            match = re.fullmatch(r'(left|right)_sensor\d+', channel)
            if match:
                hands[i] = match.group(1)
        basis = 'Released explicit hand prefixes; otherwise hand identity remains unverified.'
    groups = []
    for hand in ('left', 'right', None):
        indices = [i for i, owner in enumerate(hands) if owner == hand]
        if indices:
            groups.append(dict(label=(hand.title() + ' hand · released sensor indices' if hand else
                'Hand identity unverified · released sensor indices'), indices=indices, hand=hand))
    measured = [h for h in ('left', 'right') if h in hands]
    # Unknown ownership is not evidence that the other hand is absent.
    unmeasured = [h for h in ('left', 'right') if h not in measured] if None not in hands else []
    return dict(policy=POLICY, labels=labels, native_channel_order=channels,
        channel_hands=hands, groups=groups, measured_hands=measured,
        unmeasured_hands=unmeasured, basis=basis, evidence_url=EVIDENCE)
