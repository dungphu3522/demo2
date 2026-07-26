"""
task_colorize.py — Face Color Enhancement and Restoration (tô màu mặt)
Model: codeformer_colorization.pth

GIỚI HẠN QUAN TRỌNG (từ chính model gốc, không phải do code viết thiếu):
    - Model NÀY chỉ được huấn luyện để tô màu vùng MẶT đã crop/align 512x512.
    - Khi chạy trên ảnh thường (có nền), code sẽ tự detect + align mặt,
      tô màu riêng khuôn mặt rồi dán lại vào đúng vị trí trên ảnh gốc.
      PHẦN NỀN của ảnh sẽ KHÔNG được tô màu (vẫn đen trắng/phai màu như cũ).
    - w (fidelity weight) LUÔN cố định = 0, vì tác giả không huấn luyện
      Stage III cho model colorization -> không có tham số để chỉnh.
"""

import os
import cv2
import torch

import common


def load_models(device=None, upscale=1, face_size=512, crop_ratio=(1, 1)):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task_colorize] Dùng thiết bị: {device}")

    net = common.build_codeformer_net("colorize", device)
    face_helper = common.build_face_helper(
        device, upscale_factor=upscale, face_size=face_size, crop_ratio=crop_ratio
    )

    return {"device": device, "net": net, "face_helper": face_helper}


def process_image(models, img_path, output_path,
                   has_aligned=True, only_center_face=False, draw_box=False):
    device = models["device"]
    net = models["net"]
    face_helper = models["face_helper"]

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")

    # QUAN TRỌNG: model codeformer_colorization.pth được train với input là ảnh
    # xám (grayscale) nhân bản thành 3 kênh. Nếu đưa thẳng ảnh màu vào, model sẽ
    # gần như giữ nguyên màu gốc -> ảnh ra "không đổi" như bạn gặp phải.
    # Ép ảnh về grayscale 3 kênh trước khi detect mặt / đưa vào mạng.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    face_helper.clean_all()

    if has_aligned:
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img_resized]
    else:
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        face_helper.align_warp_face()
        if len(face_helper.cropped_faces) == 0:
            print(f"  [Cảnh báo] Không phát hiện được khuôn mặt nào trong '{img_name}'. "
                  f"Bỏ qua tô màu (chỉ model face mới tô được), lưu lại ảnh gốc.")

    for cropped_face in face_helper.cropped_faces:
        cropped_face_t = common.face_to_tensor(cropped_face, device)
        try:
            with torch.no_grad():
                # w cố định = 0, vì codeformer_colorization.pth không train Stage III (giống repo gốc)
                output = net(cropped_face_t, w=0, adain=True)[0]
                colorized_face = common.tensor_to_face(output)
            del output
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Lỗi tô màu mặt: {e}")
            colorized_face = common.tensor_to_face(cropped_face_t)
        face_helper.add_restored_face(colorized_face)

    saved_paths = []

    if not has_aligned:
        face_helper.get_inverse_affine(None)
        # upsample_img=img (ảnh GỐC dạng grayscale-3-kênh, không upscale) vì model
        # này không làm nét/phóng to, chỉ tô màu vùng mặt rồi dán lại đúng chỗ.
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=img, draw_box=draw_box)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, restored_img)
        saved_paths.append(output_path)
    else:
        os.makedirs(output_path, exist_ok=True)
        for idx, face in enumerate(face_helper.restored_faces):
            save_path = os.path.join(output_path, f"{img_name}_{idx:02d}.png")
            cv2.imwrite(save_path, face)
            saved_paths.append(save_path)

    return saved_paths