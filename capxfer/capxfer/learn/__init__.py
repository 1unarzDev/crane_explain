"""WS-C learning package. The machine is shared and heavily loaded: small CPU tensor ops are far faster with few
intra-op threads, so cap them (override with CAPXFER_C_THREADS)."""
import os as _os

import torch as _torch

_torch.set_num_threads(int(_os.environ.get("CAPXFER_C_THREADS", "2")))
