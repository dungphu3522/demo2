"""
task_inpaint.py — Face Inpainting (vá khuôn mặt bị rách/hỏng/che khuất)
Model: codeformer_inpainting.pth

CÁCH MODEL GỐC XÁC ĐỊNH VÙNG CẦN VÁ (không dùng file mask riêng):
    Bạn phải tự SƠN TRẮNG TINH (RGB = 255,255,255) đúng vùng cần vá lên
    ảnh mặt bằng Photoshop/Paint/GIMP... trước khi đưa vào đây. Code sẽ
    tự nhận diện vùng trắng tinh đó làm mask cần vá (giống hệt cách
    inference_inpainting.py gốc làm) — KHÔNG cần file mask riêng.

LƯU Ý QUAN TRỌNG:
    - Chỉ những pixel TRẮNG TUYỆT ĐỐI (255,255,255) mới được coi là vùng
      cần vá. Ảnh trắng/xám gần đúng sẽ KHÔNG được nhận ra.
    - Nếu chạy ở chế độ ảnh thường (has_aligned=False), bước align/warp
      khuôn mặt (xoay, phóng to) có thể làm mờ viền vùng đã sơn trắng do
      nội suy, khiến mask bị mất một phần. Vì vậy khuyến nghị VẼ SẴN mask
      trắng trên đúng ảnh mặt đã crop/align 512x512 rồi chạy với
      has_aligned=True để đảm bảo mask chính xác 100% — đúng như cách
      dùng gốc của tác giả (thư mục inputs/masked_faces trong repo gốc).
"""
import os
import cv2
import torch

import common


def load_models(device=None, upscale=1, face_size=512, crop_ratio=(1, 1)):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task_inpaint] Dùng thiết bị: {device}")

    net = common.build_codeformer_net("inpaint", device)
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

    if not has_aligned:
        print("  [Cảnh báo] Đang vá ảnh KHÔNG ở chế độ has_aligned=True. Bước align/warp "
              "có thể làm mờ viền vùng trắng bạn đã sơn, khiến mask vá bị thiếu. Nếu kết quả "
              "vá không chính xác, hãy tự crop/align mặt về đúng 512x512 rồi chạy lại với "
              "has_aligned=True.")

    face_helper.clean_all()

    if has_aligned:
        h, w = img.shape[:2]
        if (h, w) != (512, 512):
            print(f"  [Cảnh báo] Ảnh input has_aligned kích thước {w}x{h}, không đúng 512x512. "
                  "Sẽ resize bằng nội suy NEAREST để không làm nhòe vùng trắng tuyệt đối bạn đã "
                  "sơn, nhưng kết quả tốt nhất vẫn là tự crop/align ảnh về đúng 512x512 trước.")
            img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            # Đã đúng 512x512 -> không cần resize, giữ nguyên pixel để mask không bị đổi.
            img_resized = img
        face_helper.cropped_faces = [img_resized]
    else:
        face_helper.read_image(img)
        face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        face_helper.align_warp_face()
        if len(face_helper.cropped_faces) == 0:
            print(f"  [Cảnh báo] Không phát hiện được khuôn mặt nào trong '{img_name}'.")

    for cropped_face in face_helper.cropped_faces:
        cropped_face_t = common.face_to_tensor(cropped_face, device)
        try:
            with torch.no_grad():
                # Vùng cần vá = nơi tensor đã chuẩn hoá có tổng 3 kênh == 3,
                # tức pixel gốc là (255,255,255) trắng tuyệt đối (giống repo gốc).
                mask = torch.zeros(512, 512, device=device)
                m_ind = torch.sum(cropped_face_t[0], dim=0)
                mask[m_ind == 3] = 1.0
                mask = mask.view(1, 1, 512, 512)

                num_mask_px = int(mask.sum().item())
                print(f"  [Debug] Số pixel mask (trắng tuyệt đối 255,255,255) phát hiện được: "
                      f"{num_mask_px} / 262144")

                if num_mask_px == 0:
                    print("  [Cảnh báo] Không tìm thấy vùng sơn trắng tuyệt đối (255,255,255) nào "
                          "trong ảnh mặt -> không có gì để vá, giữ nguyên ảnh mặt. Kiểm tra lại: "
                          "(1) đã chạy với has_aligned=True chưa, (2) ảnh mask có lưu .png "
                          "(không phải .jpg) không, (3) vùng sơn có đúng RGB (255,255,255) tuyệt "
                          "đối, không bị anti-alias/blend viền không.")

                # w cố định = 1, adain=False cho inpainting (giống repo gốc)
                output = net(cropped_face_t, w=1, adain=False)[0]
                output = (1 - mask) * cropped_face_t + mask * output
                inpainted_face = common.tensor_to_face(output)
            del output
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Lỗi vá mặt: {e}")
            inpainted_face = common.tensor_to_face(cropped_face_t)
        face_helper.add_restored_face(inpainted_face)

    saved_paths = []

    if not has_aligned:
        face_helper.get_inverse_affine(None)
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