import torch
print("PyTorch Version:", torch.__version__)  # PyTorch 버전 확인
print("CUDA Available:", torch.cuda.is_available())  # CUDA 사용 가능 여부
print("PyTorch CUDA Version:", torch.version.cuda)  # PyTorch가 사용하는 CUDA 버전
print("GPU Name:", torch.cuda.get_device_name(0))  # GPU 이름 확인
