import torch

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()


torch.onnx.export(model, torch.rand(1,3,640,640), "detr-rs50.onnx", opset_version=12)