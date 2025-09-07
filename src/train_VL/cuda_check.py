import torch, os
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
print("name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

