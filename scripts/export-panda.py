# PanDA-Small (Apache-2.0 · Depth Anything V2 ViT-S 기반) → ONNX. GitHub Actions 의 try-panda 워크플로가 불러요.
#  · 코드: github.com/caozidong/PanDA (Apache-2.0) · 가중치: huggingface.co/ZidongC/PanDA panda_small.pth (모델 카드 apache-2.0)
#  · ViT-B/L 바탕(Depth Anything V2 Base/Large = CC BY-NC)은 쓰지 않아요 — Small 만.
#  · 입력 1×3×504×1008 (등장방형, ImageNet 정규화 RGB) → 출력 pred_depth 1×1×504×1008 (상대 깊이 0~1)
import sys, os, yaml, torch
sys.path.insert(0, os.path.abspath('PanDA'))
os.chdir('PanDA')
from networks.models import make  # noqa: E402

cfg = yaml.safe_load(open('config/inference/panda_large.yaml'))
cfg['model']['args']['midas_model_type'] = 'vits'
model = make(cfg['model'])
sd = torch.load(sys.argv[1], map_location='cpu')
sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
own = model.state_dict()
miss = [k for k in own if k not in sd]
print('가중치 키', len(sd), '· 모델 키', len(own), '· 빠진 키', len(miss), miss[:5])
model.load_state_dict({k: v for k, v in sd.items() if k in own}, strict=False)
model.eval()

class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        return self.m(x)['pred_depth']

x = torch.randn(1, 3, 504, 1008)
with torch.no_grad():
    y = Wrap(model)(x)
print('출력 모양', tuple(y.shape))
torch.onnx.export(Wrap(model), x, os.path.abspath('../models/panda_small.onnx'), opset_version=17, input_names=['image'], output_names=['pred_depth'])
print('ONNX 완료', os.path.getsize('../models/panda_small.onnx') // 1048576, 'MB')
