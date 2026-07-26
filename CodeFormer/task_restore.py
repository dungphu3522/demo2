"""
task_restore.py — Face Restoration (làm nét khuôn mặt, mờ/vỡ -> nét)
Model: codeformer.pth
Đây chính là logic cũ trong main.py trước đây, chỉ chuyển sang dùng
các hàm chung ở common.py để tránh lặp code với 2 tác vụ mới.
"""

import os
import cv2
import torch

import common


def load_models(device=None, upscale=2, bg_upsampler=False, face_upsample=False,
                 bg_model_name="x4plus", bg_tile=0, bg_tile_pad=32, bg_pre_pad=10,
                 half=None, face_size=512, crop_ratio=(1, 1),
                 bg_blend=1.0, bg_sharpen=0.0, final_sharpen=0.0,
                 face_pre_sharpen=0.0, face_post_sharpen=0.0):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task_restore] Dùng thiết bị: {device}")

    net = common.build_codeformer_net("restore", device)
    face_helper = common.build_face_helper(
        device, upscale_factor=upscale, face_size=face_size, crop_ratio=crop_ratio
    )

    bg_up = None
    if bg_upsampler:
        bg_up = common.build_realesrgan_with_fallback(
            device, bg_model_name, bg_tile, bg_tile_pad, bg_pre_pad, half
        )
    face_up = bg_up if face_upsample else None

    return {
        "device": device,
        "net": net,
        "face_helper": face_helper,
        "bg_upsampler": bg_up,
        "face_upsampler": face_up,
        "upscale": upscale,
        "bg_blend": max(0.0, min(1.0, bg_blend)),
        "bg_sharpen": max(0.0, bg_sharpen),
        "final_sharpen": max(0.0, final_sharpen),
        "face_pre_sharpen": max(0.0, face_pre_sharpen),
        "face_post_sharpen": max(0.0, face_post_sharpen),
    }


def process_image(models, img_path, output_path,
                   fidelity_weight=0.8, has_aligned=False,
                   only_center_face=False, draw_box=False):
    device = models["device"]
    net = models["net"]
    face_helper = models["face_helper"]
    bg_upsampler = models["bg_upsampler"]
    face_upsampler = models["face_upsampler"]
    upscale = models["upscale"]
    bg_blend = models.get("bg_blend", 1.0)
    bg_sharpen = models.get("bg_sharpen", 0.0)
    final_sharpen = models.get("final_sharpen", 0.0)
    face_pre_sharpen = models.get("face_pre_sharpen", 0.0)
    face_post_sharpen = models.get("face_post_sharpen", 0.0)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

    fidelity_weight = max(0.0, min(1.0, fidelity_weight))
    face_helper.clean_all()

    if has_aligned:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        face_helper.align_warp_face()

        try:
            for det in face_helper.det_faces:
                x1, y1, x2, y2 = det[:4]
                face_w, face_h = x2 - x1, y2 - y1
                if min(face_w, face_h) < 80:
                    print(
                        f"  [Cảnh báo] Khuôn mặt trong '{img_name}' rất nhỏ "
                        f"(~{min(face_w, face_h):.0f}px). Nguy cơ AI 'bịa' sai đặc điểm "
                        f"khuôn mặt là RẤT CAO nếu fidelity_weight < 0.8 (hiện tại: "
                        f"{fidelity_weight}). Nên tăng --fidelity_weight lên 0.9-1.0."
                    )
        except Exception:
            pass

    for cropped_face in face_helper.cropped_faces:
        if face_pre_sharpen > 0:
            cropped_face = common.unsharp_mask(cropped_face, amount=face_pre_sharpen)

        cropped_face_t = common.face_to_tensor(cropped_face, device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = common.tensor_to_face(output)
            del output
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Lỗi phục hồi mặt: {e}")
            restored_face = common.tensor_to_face(cropped_face_t)

        if face_post_sharpen > 0:
            restored_face = common.unsharp_mask(restored_face, amount=face_post_sharpen)

        face_helper.add_restored_face(restored_face)

    saved_paths = []

    if not has_aligned:
        bg_img = None
        if bg_upsampler is None:
            if upscale and upscale != 1:
                h, w = img.shape[:2]
                bg_img = cv2.resize(img, (w * upscale, h * upscale), interpolation=cv2.INTER_LANCZOS4)
            else:
                bg_img = img
        if bg_upsampler is not None:
            try:
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            except torch.cuda.OutOfMemoryError:
                print("  Hết VRAM khi nâng cấp nền, thử lại với tile nhỏ hơn...")
                torch.cuda.empty_cache()
                orig_tile = bg_upsampler.tile
                bg_upsampler.tile = 100
                try:
                    bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                finally:
                    bg_upsampler.tile = orig_tile

            if bg_img is not None:
                import numpy as np
                mean_val = float(np.mean(bg_img))
                if mean_val < 2.0:
                    print("  [Cảnh báo] Ảnh nền gần như toàn đen (nghi ngờ NaN/Inf do fp16). "
                          "Tự chuyển sang fp32 và thử lại...")
                    was_half = bg_upsampler.half
                    bg_upsampler.half = False
                    if hasattr(bg_upsampler, "model"):
                        bg_upsampler.model = bg_upsampler.model.float()
                    try:
                        bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
                    finally:
                        bg_upsampler.half = was_half

            if bg_img is not None and bg_blend < 1.0:
                h_t, w_t = bg_img.shape[:2]
                plain_up = cv2.resize(img, (w_t, h_t), interpolation=cv2.INTER_LANCZOS4)
                bg_img = cv2.addWeighted(bg_img, bg_blend, plain_up, 1.0 - bg_blend, 0).astype("uint8")

        if bg_img is not None:
            bg_img = common.unsharp_mask(bg_img, amount=bg_sharpen)

        face_helper.get_inverse_affine(None)

        if face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_box,
            )

        if final_sharpen > 0 and restored_img is not None:
            restored_img = common.unsharp_mask(restored_img, amount=final_sharpen)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, restored_img)
        saved_paths.append(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)
        for idx, restored_face in enumerate(face_helper.restored_faces):
            save_path = os.path.join(output_path, f"{img_name}_{idx:02d}.png")
            cv2.imwrite(save_path, restored_face)
            saved_paths.append(save_path)

    return saved_paths
