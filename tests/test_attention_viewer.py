import base64

import numpy as np
import pytest
import torch

from tactilelatent.attention_viewer import capture_cross_attention,cross_attention_weights,encode_weights
from tactilelatent.vjepa_spatial import SpatialFrameEncoder


def test_probabilities_match_pytorch_cross_attention():
    torch.set_num_threads(2);torch.manual_seed(45)
    attention=torch.nn.MultiheadAttention(16,4,batch_first=True).eval()
    q=torch.randn(2,3,16);memory=torch.randn(2,48,16)
    padding=torch.zeros(2,48,dtype=torch.bool);padding[:,16:32]=True
    weights=cross_attention_weights(attention,(q,memory,memory),dict(key_padding_mask=padding))
    _,expected=attention(q,memory,memory,key_padding_mask=padding,need_weights=True,average_attn_weights=False)
    torch.testing.assert_close(weights,expected,atol=1e-6,rtol=1e-6)
    assert (weights[...,16:32]==0).all()


@pytest.mark.parametrize('prediction_mode,query',[('current_frame',0),('all_frames',3)])
def test_capture_is_read_only_and_has_correct_spatial_time_camera_layout(prediction_mode,query):
    torch.manual_seed(45)
    visual=SpatialFrameEncoder(input_dim=12,width=16,heads=4,layers=2,feedforward=32,
        grid_size=2,prediction_mode=prediction_mode).eval()
    features=torch.randn(1,4,3,4,12);mask=torch.ones(1,8,3,dtype=torch.bool);mask[:,:,1:]=False
    features[:,:,1:]=float('nan')
    state={k:v.clone() for k,v in visual.state_dict().items()}
    with torch.inference_mode():before=visual.encode(features,mask)[0]
    weights=capture_cross_attention(visual,features,mask,query)
    with torch.inference_mode():after=visual.encode(features,mask)[0]
    torch.testing.assert_close(before,after,atol=0,rtol=0)
    assert weights.shape==(2,4,4,3,2,2)
    assert torch.isfinite(weights).all() and (weights[:,:,:,1:]==0).all()
    torch.testing.assert_close(weights.sum((2,3,4,5)),torch.ones(2,4))
    for key,value in visual.state_dict().items():torch.testing.assert_close(state[key],value,atol=0,rtol=0)
    assert all(p.grad is None for p in visual.parameters())
    assert all(not layer.multihead_attn._forward_pre_hooks for layer in visual.decoder.layers)
    encoded=encode_weights(weights)
    decoded=np.frombuffer(base64.b64decode(encoded['weights_b64']),dtype='<f4').reshape(encoded['shape'])
    np.testing.assert_array_equal(decoded,weights.numpy())


def test_hooks_are_removed_after_invalid_query():
    visual=SpatialFrameEncoder(input_dim=12,width=16,heads=4,layers=1,feedforward=32,
        grid_size=2,prediction_mode='current_frame').eval()
    with pytest.raises(ValueError,match='query'):
        capture_cross_attention(visual,torch.zeros(1,4,3,4,12),torch.ones(1,8,3,dtype=torch.bool),7)
    assert not visual.decoder.layers[0].multihead_attn._forward_pre_hooks


def test_capture_rejects_training_mode():
    visual=SpatialFrameEncoder(width=16,heads=4)
    with pytest.raises(ValueError,match='eval'):
        capture_cross_attention(visual,torch.empty(1),torch.empty(1))
