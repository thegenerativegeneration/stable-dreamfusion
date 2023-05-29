FROM wawa9000/stable-dreamfusion


ADD https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth /home/ph/.cache/torch/hub/checkpoints/jx_vit_base_resnet50_384-9fd3c705.pth
