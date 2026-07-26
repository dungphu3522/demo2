"""
common.py
=========
Nơi TẬP TRUNG mọi thứ dùng chung cho cả 3 tác vụ CodeFormer:
    - Face Restoration              (codeformer.pth)
    - Face Color Enhancement/Colorization  (codeformer_colorization.pth)
    - Face Inpainting                (codeformer_inpainting.pth)

3 file task_restore.py / task_colorize.py / task_inpaint.py chỉ chứa
logic RIÊNG của từng tác vụ, còn mọi thứ chung (đường dẫn model, kiểm
tra file, dựng mạng CodeFormer đúng kiến trúc, dựng face_helper,
Real-ESRGAN, unsharp mask, chuyển ảnh <-> tensor...) đều nằm ở đây để
tránh lặp code và tránh copy sai cấu hình giữa các model.

LƯU Ý QUAN TRỌNG (rất dễ nhầm giữa các model):
    - codeformer.pth (restore):        codebook_size=1024, connect_list có '256'
    - codeformer_colorization.pth:     codebook_size=1024, connect_list KHÔNG có '256'
    - codeformer_inpainting.pth:       codebook_size=512  (khác!), connect_list KHÔNG có '256'
Dùng sai codebook_size hoặc connect_list sẽ khiến load_state_dict lỗi
hoặc chạy được nhưng ra kết quả sai/vỡ ảnh.
"""

import os
import torch
import numpy as np
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

# ---- Đường dẫn, luôn tính theo vị trí của chính file này ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(_THIS_DIR, "weights")

RESTORE_WEIGHT = os.path.join(WEIGHTS_DIR, "CodeFormer", "codeformer.pth")
COLORIZE_WEIGHT = os.path.join(WEIGHTS_DIR, "CodeFormer", "codeformer_colorization.pth")
INPAINT_WEIGHT = os.path.join(WEIGHTS_DIR, "CodeFormer", "codeformer_inpainting.pth")

DETECTION_WEIGHT = os.path.join(WEIGHTS_DIR, "facelib", "detection_Resnet50_Final.pth")
PARSING_WEIGHT = os.path.join(WEIGHTS_DIR, "facelib", "parsing_parsenet.pth")

REALESRGAN_X2_WEIGHT = os.path.join(WEIGHTS_DIR, "realesrgan", "RealESRGAN_x2plus.pth")
REALESRGAN_X4_WEIGHT = os.path.join(WEIGHTS_DIR, "realesrgan", "RealESRGAN_x4plus.pth")

# Dung lượng tối thiểu hợp lệ (bytes) - để bắt lỗi tải thiếu/sai file sớm.
# Số liệu tham khảo từ trang Release chính thức (Assets), lấy dư an toàn ~30%.
_MIN_SIZE_BYTES = {
    RESTORE_WEIGHT: 60 * 1024 * 1024,     # thật ~359MB
    COLORIZE_WEIGHT: 100 * 1024 * 1024,   # thật ~355MB
    INPAINT_WEIGHT: 100 * 1024 * 1024,    # thật ~354MB
    DETECTION_WEIGHT: 100 * 1024 * 1024,  # thật ~104MB
    PARSING_WEIGHT: 80 * 1024 * 1024,     # thật ~81.4MB
    REALESRGAN_X2_WEIGHT: 60 * 1024 * 1024,
    REALESRGAN_X4_WEIGHT: 60 * 1024 * 1024,
}


def check_weight(path, name=None):
    """Kiểm tra file model có tồn tại và không bị tải thiếu/sai."""
    name = name or os.path.basename(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Không tìm thấy '{name}' tại: {path}\n"
            f"Bạn cần tự tải file này (từ Assets của Release chính thức) và đặt "
            f"đúng vị trí trước khi chạy."
        )
    min_size = _MIN_SIZE_BYTES.get(path)
    actual_size = os.path.getsize(path)
    if min_size and actual_size < min_size:
        raise ValueError(
            f"File '{name}' tại {path} chỉ nặng {actual_size / (1024*1024):.1f} MB, "
            f"nhỏ hơn nhiều so với mức tối thiểu hợp lệ (~{min_size / (1024*1024):.0f} MB).\n"
            f"=> File gần như chắc chắn bị tải THIẾU hoặc SAI. Hãy xoá và tải lại từ "
            f"đúng nguồn (Assets trong trang Release)."
        )


# ---- Cấu hình kiến trúc CodeFormer cho từng tác vụ (xem ghi chú đầu file) ----
NET_CONFIGS = {
    "restore": dict(
        weight=RESTORE_WEIGHT, dim_embd=512, codebook_size=1024,
        n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"],
    ),
    "colorize": dict(
        weight=COLORIZE_WEIGHT, dim_embd=512, codebook_size=1024,
        n_head=8, n_layers=9, connect_list=["32", "64", "128"],
    ),
    "inpaint": dict(
        weight=INPAINT_WEIGHT, dim_embd=512, codebook_size=512,
        n_head=8, n_layers=9, connect_list=["32", "64", "128"],
    ),
}


def build_codeformer_net(task, device):
    """Dựng đúng kiến trúc CodeFormer và load đúng weight cho từng tác vụ."""
    if task not in NET_CONFIGS:
        raise ValueError(f"task không hợp lệ: {task} (chỉ nhận {list(NET_CONFIGS)})")
    cfg = NET_CONFIGS[task]
    check_weight(cfg["weight"])

    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=cfg["dim_embd"],
        codebook_size=cfg["codebook_size"],
        n_head=cfg["n_head"],
        n_layers=cfg["n_layers"],
        connect_list=cfg["connect_list"],
    ).to(device)

    ckpt = torch.load(cfg["weight"], map_location=device)
    net.load_state_dict(ckpt["params_ema"] if "params_ema" in ckpt else ckpt)
    net.eval()
    return net


def build_face_helper(device, upscale_factor=1, face_size=512, crop_ratio=(1, 1)):
    check_weight(DETECTION_WEIGHT)
    check_weight(PARSING_WEIGHT)
    return FaceRestoreHelper(
        upscale_factor=upscale_factor,
        face_size=face_size,
        crop_ratio=crop_ratio,
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )


def build_realesrgan(device, model_name="x4plus", tile=400, tile_pad=32,
                      pre_pad=10, half=None):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    if model_name == "x4plus":
        weight_path = REALESRGAN_X4_WEIGHT
        netscale = 4
    elif model_name == "x2plus":
        weight_path = REALESRGAN_X2_WEIGHT
        netscale = 2
    else:
        raise ValueError(f"model_name không hợp lệ: {model_name} (chỉ nhận 'x2plus' hoặc 'x4plus')")

    check_weight(weight_path, os.path.basename(weight_path))

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_block=23, num_grow_ch=32, scale=netscale)

    if half is None:
        half = device.type == "cuda"

    return RealESRGANer(
        scale=netscale, model_path=weight_path, model=model,
        tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=half, device=device,
    )


def build_realesrgan_with_fallback(device, model_name, tile, tile_pad, pre_pad, half):
    try:
        return build_realesrgan(device, model_name, tile, tile_pad, pre_pad, half)
    except torch.cuda.OutOfMemoryError:
        fallback_tile = 200 if (tile == 0 or tile > 200) else 100
        print(f"[common.py] Hết VRAM với tile={tile}, tự giảm xuống tile={fallback_tile} và thử lại...")
        torch.cuda.empty_cache()
        return build_realesrgan(device, model_name, fallback_tile, tile_pad, pre_pad, half)


def unsharp_mask(img, amount=0.8, radius=3):
    """Unsharp mask đơn giản bằng OpenCV. amount<=0 -> trả nguyên ảnh."""
    if amount <= 0:
        return img
    import cv2
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype("uint8")


def face_to_tensor(face_bgr, device):
    """Ảnh mặt BGR uint8 (512x512) -> tensor chuẩn hoá, sẵn sàng đưa vào mạng."""
    t = img2tensor(face_bgr / 255.0, bgr2rgb=True, float32=True)
    normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return t.unsqueeze(0).to(device)


def tensor_to_face(t):
    """Tensor output của mạng -> ảnh BGR uint8."""
    img = tensor2img(t, rgb2bgr=True, min_max=(-1, 1))
    return img.astype("uint8")
