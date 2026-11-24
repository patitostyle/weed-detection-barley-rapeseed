"""Matched source split, normalizations and evaluation support for DeepLabv3.

ImageNet pretraining applies to the ResNet-50 backbone only, not the decoder.
No ImageNet weights are used in the randomly initialized comparison.
"""
import argparse
import torch
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
import run_revision as r

torch.hub.set_dir(str(r.HERE/'torch_cache'))

class DeepLab(torch.nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        self.net=deeplabv3_resnet50(weights=None,
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            num_classes=3,aux_loss=False)
        old=self.net.backbone.conv1
        new=torch.nn.Conv2d(8,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
        if pretrained:
            with torch.no_grad():
                new.weight[:,:3].copy_(old.weight)
                new.weight[:,3:].copy_(old.weight.mean(1,keepdim=True).repeat(1,5,1,1))
                new.weight.mul_(3/8)
        self.net.backbone.conv1=new
    def forward(self,x):return self.net(x)['out']

def train():
    m=r.prepare();stats=r.source_stats(m['source']['training'])
    for norm in r.NORMS:
        tr=r.Data(m['source']['training'],norm,stats,True)
        val=r.Data(m['source']['validation'],norm,stats)
        for pretrained in [False,True]:
            for seed in r.SOURCE_SEEDS:
                path=r.OUT/f"deeplab_{norm}_{'imagenet' if pretrained else 'random'}_seed{seed}"
                if (path/'complete.json').exists():continue
                r.legacy.seed_all(seed)
                model=DeepLab(pretrained)
                r.train_one(path,tr,val,seed,'source_deeplab',model=model)
        del tr,val

def check():
    r.legacy.seed_all(42)
    model=DeepLab(True)
    model.eval()
    with torch.no_grad():out=model(torch.zeros(1,8,64,64))
    assert out.shape==(1,3,64,64)
    r.dump(r.OUT/'deeplab_initialization.json',{
        'backbone':'ResNet-50','pretraining':'ImageNet1K V2','decoder':'random',
        'weight_url':ResNet50_Weights.IMAGENET1K_V2.url,
        'input_adaptation':'RGB weights copied; five extra channels initialized with RGB weight mean; all multiplied by 3/8',
        'normalization':'same source-trained scaling or patch scaling as comparison U-Net; no claim of architecture-wide superiority',
        'batch_handling':'drop final incomplete training batch to avoid singleton ASPP BatchNorm',
        'forward_check':list(out.shape)})
    print('Pretrained 8-channel DeepLab forward check passed',flush=True)

if __name__=='__main__':
    raise SystemExit('This file provides model/training helpers. Use reproduce_target.py, staged_revision.py or the documented benchmark scripts; the superseded full-grid entry point is disabled.')
