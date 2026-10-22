"""Visual 32D -> source-specific AE code; frozen marker teachers supply targets."""
import torch

from .aligned_frames import file_hash
from .marker_autoencoder import load_marker_autoencoder
from .marker_targets import MARKER_SOURCES
from .train_vjepa_spatial import LiveTeachers
from .vjepa_spatial import JointSpatialVJEPA

MODEL_TYPE='joint_vjepa21_spatial_marker_codes_v1'


class MarkerCodeVJEPA(JointSpatialVJEPA):
    def __init__(self,**kwargs):
        if not set(MARKER_SOURCES)&set(kwargs['sensor_code_dims']):raise ValueError('Missing marker-code adapters')
        super().__init__(**kwargs);self.model_type=MODEL_TYPE

    def forward(self,features,view_mask,source,sample_latent=None,marker_reference=None,marker_surface=None):
        return super().forward(features,view_mask,source,sample_latent=sample_latent)


def model_from_state(state):
    if state.get('model_type')!=MODEL_TYPE:raise ValueError('Expected marker-code alignment model')
    model=MarkerCodeVJEPA(**state['model_config']);model.load_state_dict(state['model'],strict=True);return model


class MixedMarkerTeachers(LiveTeachers):
    def __init__(self,registry,device,chunk_frames=32):
        marker={s:v for s,v in registry['sources'].items() if v.get('representation')=='marker_displacement'}
        native={s:v for s,v in registry['sources'].items() if s not in marker}
        if not marker or not set(marker)<=set(MARKER_SOURCES):raise ValueError('Expected selected frozen marker AEs')
        super().__init__({**registry,'sources':native},device,chunk_frames=1)
        self.marker_sources=set(marker);self.marker_chunk_frames=chunk_frames
        for source,info in marker.items():
            if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Marker teacher provenance mismatch')
            self.models[source]=load_marker_autoencoder(info['checkpoint'],device)[0]

    @torch.no_grad()
    def supervision(self,original,model):
        source=original['source'][0]
        if 'code_target' in original:
            if source not in model.normalization:raise ValueError('Cached codes require a sensor adapter')
            if original['code_target'].shape[-1]!=model.normalization[source].mean.numel():raise ValueError('Cached code dimension differs')
            return dict(code_target=original['code_target'].to(self.device),code_valid=original['code_valid'].to(self.device))
        if source not in self.marker_sources:return super().supervision(original,model)
        y=original['marker_target'].to(self.device).float();valid=original['marker_valid'].to(self.device)
        b,t,n,_=y.shape
        ref=original['marker_reference'].to(self.device)[:,None].expand(-1,t,-1,-1).reshape(b*t,n,2)
        surface=original['marker_surface'].to(self.device)[:,None].expand(-1,t,-1).reshape(b*t,n)
        values=[]
        with torch.autocast(self.device.type,enabled=False):
            for start in range(0,b*t,self.marker_chunk_frames):
                sl=slice(start,start+self.marker_chunk_frames)
                values.append(self.models[source].encode(y.reshape(b*t,n,2)[sl],valid.reshape(b*t,n)[sl],ref[sl],surface[sl]))
        raw=torch.cat(values).reshape(b,t,-1);norm=model.normalization[source]
        return dict(code_target=(raw-norm.mean)/norm.std,code_valid=valid.any(-1))

    @torch.no_grad()
    def decode_marker(self,source,raw_code,batch):
        b,t,c=raw_code.shape;ref=batch['marker_reference'].to(self.device);surface=batch['marker_surface'].to(self.device);n=ref.shape[1]
        with torch.autocast(self.device.type,enabled=False):
            value=self.models[source].decode(raw_code.float().reshape(b*t,c),
                ref[:,None].expand(-1,t,-1,-1).reshape(b*t,n,2),surface[:,None].expand(-1,t,-1).reshape(b*t,n))
        return value.reshape(b,t,n,2)

    @torch.no_grad()
    def decode(self,source,raw_code):
        if source in self.marker_sources:raise ValueError('Marker decoding requires static point geometry')
        leading=raw_code.shape[:-1];ae=self.models[source]
        with torch.autocast(self.device.type,enabled=False):value=ae.decode(raw_code.float().reshape(-1,raw_code.shape[-1]))
        return value.reshape(*leading,*value.shape[1:])
