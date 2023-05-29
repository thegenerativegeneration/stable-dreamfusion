import sys

import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *

import gradio as gr
import gc
from params import parser
from main import prepare

opt = parser.parse_args()

opt.default_zero123_w = 1

opt.exp_start_iter = opt.exp_start_iter or 0
opt.exp_end_iter = opt.exp_end_iter or opt.iters

# Parameters for low memory
opt.O = True
opt.vram_O = True

opt.dir_text = True
# opt.lambda_entropy = 1e-4
# opt.lambda_opacity = 0



print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] loading models..')

train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

print(f'[INFO] everything loaded!')

trainer = None
model = None

opt.test = True

# define UI

with gr.Blocks(css=".gradio-container {max-width: 512px; margin: auto;}") as demo:
    # title
    gr.Markdown('[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) Text-to-3D Example')

    # inputs
    prompt = gr.Textbox(label="Prompt", max_lines=1, value="a DSLR photo of a koi fish")
    iters = gr.Slider(label="Iters", minimum=1000, maximum=20000, value=5000, step=100)
    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, randomize=True)
    # allow multiple selections for guidance
    guidance_type = gr.CheckboxGroup(label="Guidance", choices=["SD", "IF", "zero123", "clip"],
                                      default=["SD"])
    button = gr.Button('Generate')

    # outputs
    image = gr.Image(label="image", visible=True)
    video = gr.Video(label="video", visible=False)
    logs = gr.Textbox(label="logging")


    # gradio main func
    def submit(text, iters, seed, guidance_type):

        global trainer, model, opt


        # clean up
        if trainer is not None:
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print('[INFO] clean up!')

        # prepare
        # seed
        opt.seed = seed
        opt.text = text
        opt.iters = iters
        opt.guidance = guidance_type
        if 'IF' in opt.guidance:
            opt.IF = True
        seed_everything(seed)

        model, _, opt = prepare(opt)

        guidance = nn.ModuleDict()

        if 'SD' in opt.guidance:
            from guidance.sd_utils import StableDiffusion
            guidance['SD'] = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key, opt.t_range)

        if 'IF' in opt.guidance:
            from guidance.if_utils import IF
            guidance['IF'] = IF(device, opt.vram_O, opt.t_range)

        if 'zero123' in opt.guidance:
            from guidance.zero123_utils import Zero123
            guidance['zero123'] = Zero123(device=device, fp16=opt.fp16, config=opt.zero123_config, ckpt=opt.zero123_ckpt, vram_O=opt.vram_O, t_range=opt.t_range, opt=opt)

        if 'clip' in opt.guidance:
            from guidance.clip_utils import CLIP
            guidance['clip'] = CLIP(device)

        # simply reload everything...
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer,
                                                                  lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer(' '.join(sys.argv), 'df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer,
                          ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt,
                          scheduler_update_every_step=True)

        # train (every ep only contain 8 steps, so we can get some vis every ~10s)
        STEPS = 8
        max_epochs = np.ceil(opt.iters / STEPS).astype(np.int32)

        # we have to get the explicit training loop out here to yield progressive results...
        loader = iter(valid_loader)

        start_t = time.time()

        for epoch in range(max_epochs):

            trainer.train_gui(train_loader, step=STEPS)

            # manual test and get intermediate results
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(valid_loader)
                data = next(loader)

            trainer.model.eval()

            if trainer.ema is not None:
                trainer.ema.store()
                trainer.ema.copy_to()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=trainer.fp16):
                    preds, preds_depth, _ = trainer.test_step(data, perturb=False)

            if trainer.ema is not None:
                trainer.ema.restore()

            pred = preds[0].detach().cpu().numpy()
            # pred_depth = preds_depth[0].detach().cpu().numpy()

            pred = (pred * 255).astype(np.uint8)

            yield {
                image: gr.update(value=pred, visible=True),
                video: gr.update(visible=False),
                logs: f"training iters: {epoch * STEPS} / {iters}, lr: {trainer.optimizer.param_groups[0]['lr']:.6f}",
            }

        # test
        trainer.test(test_loader)

        results = glob.glob(os.path.join(opt.workspace, 'results', '*rgb*.mp4'))
        assert results is not None, "cannot retrieve results!"
        results.sort(key=lambda x: os.path.getmtime(x))  # sort by mtime

        end_t = time.time()

        yield {
            image: gr.update(visible=False),
            video: gr.update(value=results[-1], visible=True),
            logs: f"Generation Finished in {(end_t - start_t) / 60:.4f} minutes!",
        }


    button.click(
        submit,
        [prompt, iters, seed, guidance_type],
        [image, video, logs]
    )

# concurrency_count: only allow ONE running progress, else GPU will OOM.
demo.queue(concurrency_count=1)

demo.launch()