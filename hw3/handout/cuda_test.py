import torch
print(torch.cuda.is_available())  # True여야 함
print(torch.version.cuda)         # CUDA 버전 확인
print(torch.backends.cudnn.enabled)  # True여야 함
