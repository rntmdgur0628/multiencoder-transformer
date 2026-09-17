import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from msr.data import AudioManifest
from msr.config import MOISES_SOURCES
from bsr_probe.pilot import evaluate_arm, make_schedule, paired_comparison, read_crop, sw_loss, train_arm
from bsr_probe.baseline import load_track
from bsr_probe.sw_adapter import AdapterConfig, FrozenSW, Injection, state_fingerprint


class TinyParent(nn.Module):
    """Same hook/clock/downstream interface; no external dependency."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_torch_checkpoint = False
        self.audio_channels = 2
        self.stft_kwargs = {"n_fft": cfg.n_fft, "win_length": cfg.n_fft, "hop_length": cfg.hop}
        self.band_split = nn.Identity()
        self.layers = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(cfg.base_dim, cfg.base_dim), nn.Tanh())])])
        self.mask_estimators = nn.ModuleList([nn.Linear(cfg.base_dim, 12)])
        self.projection = nn.Linear(2*cfg.n_fft, cfg.base_dim)

    def forward(self, x):
        frames = F.pad(x, (self.cfg.n_fft//2, self.cfg.n_fft//2), mode="reflect").unfold(-1, self.cfg.n_fft, self.cfg.hop)
        tokens = self.projection(frames.permute(0,2,1,3).flatten(2))[:, :, None].repeat(1,1,len(self.cfg.bands),1)
        tokens = self.band_split(tokens)
        for blocks in self.layers:
            for block in blocks:
                tokens = block(tokens)
        signal = self.mask_estimators[0](tokens.mean(2)).transpose(1,2)
        return F.interpolate(signal, size=x.shape[-1], mode="linear", align_corners=False).reshape(x.shape[0],6,2,-1)


class PilotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cfg = AdapterConfig((8,8,8,9), 64, 16, 16, short=32, width=8, heads=2)

    def test_exact_parameter_match_and_initialization(self):
        a, b = Injection(self.cfg, "tc"), Injection(self.cfg, "stft")
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        for (name, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
            self.assertEqual(name, nb)
            self.assertTrue(torch.equal(pa,pb),name)
        self.assertFalse(torch.equal(b.memory.windows[0], b.memory.windows[1]))
        x=torch.randn(1,2,128)
        self.assertEqual(a.memory(x).shape,(1,9,5,8))

    def test_zero_gate_gradient_checkpoint_and_parent_freeze(self):
        for arm in ("tc","stft"):
            base=TinyParent(self.cfg)
            plain=FrozenSW(copy.deepcopy(base),self.cfg,arm,checkpoint_blocks=False)
            cp=FrozenSW(copy.deepcopy(base),self.cfg,arm,checkpoint_blocks=True)
            x=torch.randn(1,2,128)
            with torch.no_grad():
                torch.testing.assert_close(cp(x),base(x),atol=0,rtol=0)
                cp.adapter.gate.fill_(0.01)
                plain.adapter.gate.fill_(0.01)
            fingerprint=state_fingerprint(cp.backbone)
            cp.train(); plain.train()
            cp(x).square().mean().backward(); plain(x).square().mean().backward()
            for a,b in zip(cp.adapter.parameters(),plain.adapter.parameters()):
                torch.testing.assert_close(a.grad,b.grad)
            self.assertGreater(sum(float(p.grad.norm()) for p in cp.adapter.memory.parameters() if p.grad is not None),0)
            self.assertTrue(all(p.grad is None for p in cp.backbone.parameters()))
            self.assertFalse(cp.backbone.training)
            self.assertEqual(state_fingerprint(cp.backbone),fingerprint)
            with torch.no_grad(): torch.testing.assert_close(cp(x,gate_off=True),base(x),rtol=0,atol=0)

    def test_schedule_cycles_and_model_rng_independence(self):
        tracks=[{"id":str(i),"split":"train","length":48000*20} for i in range(5)]
        a=make_schedule(tracks,12,9,44100,48000,44100,512)
        torch.manual_seed(777)
        b=make_schedule(list(reversed(tracks)),12,9,44100,48000,44100,512)
        self.assertEqual(a,b)
        self.assertEqual(len({x['id'] for x in a[:5]}),5)
        self.assertTrue(all(x['start']%512==0 for x in a))
        tracks[0]['split']='validation'
        with self.assertRaises(ValueError):make_schedule(tracks,2,9,44100,48000,44100,512)

    def fixture(self, root, length=48000):
        x=np.random.default_rng(4).normal(size=(length,2)).astype(np.float32)*0.02
        sf.write(root/'a.wav',x,48000,subtype='FLOAT')
        track={"id":"song","group":"song","split":"train","length":length,
               "targets":{s:['a.wav'] if s=='vocals' else [] for s in MOISES_SOURCES}}
        (root/'manifest.json').write_text(json.dumps({"schema_version":1,"task":"moises6","sample_rate":48000,
            "sources":list(MOISES_SOURCES),"tracks":[track]}))
        return AudioManifest(root/'manifest.json',audit_audio=False),track,x

    def test_crop_resampling_equals_full_track(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest,track,x=self.fixture(Path(temp),96031)
            full=resample_poly(x,147,160,axis=0)
            for start,count in [(0,3000),(512,8192),(25088,17920),(len(full)-4096,4096)]:
                mix,y=read_crop(manifest,track,start,count)
                np.testing.assert_allclose(mix[0].numpy().T,full[start:start+count],rtol=0,atol=1e-7)
                self.assertTrue(torch.equal(mix,y[:,0]))
                self.assertEqual(float(y[:,1:].abs().sum()),0)

    def test_loss_and_three_update_training_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,track,_=self.fixture(root)
            config={"training":{"instruments":list(MOISES_SOURCES)},"model":{
                "stft_n_fft":64,"multi_stft_resolutions_window_sizes":[64,32],
                "multi_stft_hop_size":16,"multi_stft_normalized":False,"multi_stft_resolution_loss_weight":1.}}
            settings={"precision":"fp32","learning_rate":1e-3,"gradient_clip":1.0}
            schedule=make_schedule([track],3,9,44100,48000,128,16)
            for arm in ('tc','stft'):
                model=FrozenSW(TinyParent(self.cfg),self.cfg,arm)
                result,hashes=train_arm(model,config,manifest,schedule,settings,root/arm,torch.device('cpu'),{})
                self.assertTrue(result['memory_gradient_seen'])
                self.assertTrue(result['parent_unchanged'])
                saved=torch.load(root/arm/'last.pt',weights_only=True)
                self.assertEqual(saved['step'],3)
                self.assertEqual(len(hashes),1)
            p=torch.randn(1,6,2,128,requires_grad=True)
            loss,_=sw_loss(p,p.detach(),config)
            self.assertEqual(float(loss.detach()),0)

    def test_evaluation_preserves_source_order_and_checks_input_hashes(self):
        class VocalOnly(nn.Module):
            def forward(self,x):
                result=torch.zeros(x.shape[0],6,2,x.shape[-1])
                result[:,3]=x
                return result
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,track,_=self.fixture(root,1000)
            track['split']='validation'
            manifest.tracks[0]['split']='validation'
            _,_,data=load_track(manifest,track,44100)
            config={'audio':{'chunk_size':256},'inference':{'num_overlap':2},
                    'training':{'instruments':['bass','drums','other','vocals','guitar','piano']}}
            base=[{'id':'song','audio_sha256':data['audio_sha256']}]
            rows=evaluate_arm(VocalOnly(),manifest,config,base,root/'evaluation',torch.device('cpu'),'fp32')
            self.assertGreater(rows[0]['sources']['vocals']['si_sdr_db'],100)
            self.assertEqual(rows[0]['sources']['bass']['status'],'silent_target')
            self.assertTrue((root/'evaluation/summary.json').is_file())
            base[0]['audio_sha256']={}
            with self.assertRaisesRegex(ValueError,'differs from B0'):
                evaluate_arm(VocalOnly(),manifest,config,base,root/'mismatch',torch.device('cpu'),'fp32')

    def test_pairing_does_not_hide_degenerate_predictions(self):
        a=[{'id':'a','sources':{'piano':{'status':'ok','si_sdr_db':1.,'snr_db':2.,'prediction_rms':.1}}}]
        b=copy.deepcopy(a);b[0]['sources']['piano']['si_sdr_db']=None
        b[0]['sources']['piano']['status']='degenerate_prediction'
        delta=paired_comparison(a,b,['piano'])['piano']
        self.assertEqual(delta['si_sdr_db']['invalid_active_ids'],['a'])
        self.assertIsNone(delta['si_sdr_db']['mean_delta_db'])
        self.assertEqual(delta['snr_db']['mean_delta_db'],0)


if __name__=='__main__':
    unittest.main()
