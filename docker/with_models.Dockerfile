FROM python:3.10.11-slim-buster AS models

RUN apt-get update && apt-get install -y wget git

ADD https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt 105000.ckpt
ADD https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth jx_vit_base_resnet50_384-9fd3c705.pth
RUN gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
RUN gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt

FROM wawa9000/stable-dreamfusion

COPY --from=models 105000.ckpt /app/stable-dreamfusion/pretrained/zero123/105000.ckpt
COPY --from=models jx_vit_base_resnet50_384-9fd3c705.pth ~/.cache/torch/hub/checkpoints/jx_vit_base_resnet50_384-9fd3c705.pth
COPY --from=models omnidata_dpt_depth_v2.ckpt /app/stable-dreamfusion/pretrained/omnidata/omnidata_dpt_depth_v2.ckpt
COPY --from=models omnidata_dpt_normal_v2.ckpt /app/stable-dreamfusion/pretrained/omnidata/omnidata_dpt_normal_v2.ckpt