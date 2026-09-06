"""Fail-fast CUDA diagnostic for the Nino RL environment."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print(
            "LỖI: chưa cài PyTorch. Hãy tạo .venv và cài requirements.txt theo README_VI.md.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime của PyTorch: {torch.version.cuda}")
    print(f"CUDA khả dụng: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print(
            "LỖI: PyTorch không nhìn thấy GPU NVIDIA. Không bắt đầu train bằng CPU ngoài ý muốn.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"Kiến trúc trong wheel: {', '.join(torch.cuda.get_arch_list())}")
    tensor = torch.randn((1024, 1024), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    print(f"Phép nhân CUDA kiểm tra: OK ({result.device})")


if __name__ == "__main__":
    main()
