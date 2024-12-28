import os
from config import Config

# Actively modify the third-party library: IMAGE 
import sys
from PIL import Image
try:
    from PIL import Resampling
    # 检查是否存在ANTIALIAS属性，如果不存在则添加别名
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Resampling.LANCZOS
except ImportError:
    pass

# create experiment config containing all hyperparameters
cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)

# start time integration
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        model.initialize()
    else:
        model.step()

    model.write_output(output_folder)
