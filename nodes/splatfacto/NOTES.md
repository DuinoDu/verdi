# splatfacto: notes

## Upstream use
- Same stack as SimFoundry `install_nerfstudio.sh`: nerfstudio 1.1.5 + gsplat
  1.5.3 (overrides nerfstudio's hard gsplat==1.4.0 pin), torch 2.8.0 (PyPI =
  cu128, needed for sm_120). gsplat is JIT-compiled once by a setup command
  into `{venv}/torch_extensions` (`TORCH_CUDA_ARCH_LIST={cuda_arch}`).
- SimFoundry `patches/splatfacto_depth_loss.patch` is applied to the installed
  nerfstudio by a setup command (idempotent) and enabled per run by env vars.
- entry.py mirrors SimFoundry auto_bg `5_train_bg_splat.py`: transforms.json
  from known poses (OpenCV -> OpenGL camera axes), no recentring/rescaling,
  `ns-train ... nerfstudio-data`, then `ns-export gaussian-splat`; the splat
  is then rendered back at every input pose for a PSNR fit check.

## Pitfalls
- torch JIT extension loading needs `ninja` on PATH even for a cached build
  (H20 failure "Ninja is required"): entry.py prepends the venv bin and
  `$CUDA_HOME/bin` to PATH for itself and the ns-train/ns-export subprocesses.
- splatfacto builds torchmetrics LPIPS (torchvision alexnet) at model init,
  which downloads `alexnet-owt-7be5be79.pth` from download.pytorch.org: now a
  pinned `[weights.alexnet]` url+sha256, exposed through a per-run TORCH_HOME.
- splatfacto trains at 1/4 resolution until step 3000 and 1/2 until 6000
  (num_downscales=2, resolution_schedule=3000): 2000-step runs look blurry
  and speckled at full resolution (PSNR ~18.7). The test uses 8000 steps.
- The bedroom fixture is dynamic (two children jumping): the static
  background is reconstructed sharply, the children are averaged, so PSNR
  is bounded (~20.5-22.8 dB per frame on the RTX 5090 host at 8000 steps). Camera
  optimisation off / depth loss off did not change this (18.3-18.8 at 2000).

## the RTX 5090 host (RTX 5090, sm_120)
- torch 2.7.1 -> 2.8.0, torchvision 0.23.0; `{cuda_home}` / `{cuda_arch}`;
  `sm_120` in gpu.arch; uv.lock re-locked against aliyun.
- fpsample (nerfstudio dep) builds from sdist and needs g++ (provided by
  `~/pdebug_home/bin` on the RTX 5090 host).
- Peak VRAM ~15 GB for splatfacto-big at 960x540 x 6 frames.
