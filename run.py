import os
import sys
import glob
import shutil
import tempfile
import subprocess
import argparse
import torch
from tkinter import filedialog, Tk

# ---- Cho phép import main.py bên trong CodeFormer/ như 1 module ----
CODEFORMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CodeFormer")
if CODEFORMER_DIR not in sys.path:
    sys.path.insert(0, CODEFORMER_DIR)

import main as hub  # main.py bên trong CodeFormer/ (trung tâm điều phối 3 tác vụ)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

TASK_MENU = [
    ("restore", "Face Restoration - Làm nét khuôn mặt mờ/vỡ"),
    ("colorize", "Face Color Enhancement and Restoration - Tô màu ảnh đen trắng/phai màu"),
    ("inpaint", "Face Inpainting - Vá khuôn mặt bị rách/hỏng/che khuất"),
]


def select_task_interactive():
    print("\n" + "=" * 60)
    print("Bạn muốn làm gì với ảnh này?")
    for idx, (key, label) in enumerate(TASK_MENU, start=1):
        print(f"  {idx}. {label}")
    print(f"  0. Kết thúc")
    print("=" * 60)
    while True:
        choice = input("Chọn số: ").strip()
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(TASK_MENU):
            return TASK_MENU[int(choice) - 1][0]
        print("Lựa chọn không hợp lệ, thử lại.")


def select_input_path():
    """Hiển thị hộp thoại chọn file ảnh hoặc video."""
    root = Tk()
    root.withdraw()
    file = filedialog.askopenfilename(
        title="Chọn file ảnh hoặc video",
        filetypes=[
            ("Image/Video files", "*.jpg *.jpeg *.png *.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
            ("Image files", "*.jpg *.jpeg *.png"),
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v"),
            ("All files", "*.*"),
        ]
    )
    root.destroy()
    return file or None


def collect_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    return sorted(
        glob.glob(os.path.join(input_path, "*.jpg"))
        + glob.glob(os.path.join(input_path, "*.jpeg"))
        + glob.glob(os.path.join(input_path, "*.png"))
    )


def is_video_file(path):
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


# ============================================================
#          XỬ LÝ VIDEO (chỉ hỗ trợ cho tác vụ "restore")
# ============================================================

def process_video_opencv(video_path, output_path, models, args):
    import cv2

    print("[run.py] Dùng đường xử lý OpenCV (không cần ffmpeg.exe).")
    print("[run.py] LƯU Ý: video kết quả sẽ KHÔNG có âm thanh (OpenCV không xử lý audio).")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[run.py] FPS: {fps:.3f} | Tổng số khung hình (ước tính): {total}")

    ext = os.path.splitext(output_path)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        os.makedirs(output_path, exist_ok=True)
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_path, f"{name}_restored.mp4")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    work_dir = tempfile.mkdtemp(prefix="codeformer_video_cv2_")
    writer = None
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1

            tmp_in = os.path.join(work_dir, "in_frame.png")
            tmp_out = os.path.join(work_dir, "out_frame.png")
            cv2.imwrite(tmp_in, frame)

            try:
                hub.process_image(
                    "restore", models, tmp_in, tmp_out,
                    fidelity_weight=args.fidelity_weight,
                    has_aligned=True,
                    only_center_face=args.only_center_face,
                    draw_box=args.draw_box,
                )
                out_frame = cv2.imread(tmp_out)
            except Exception as e:
                print(f"  Lỗi khung hình {idx}: {e} -> dùng khung gốc")
                out_frame = frame

            if writer is None:
                h, w = out_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(
                        "Không tạo được VideoWriter. Thử đổi đuôi file đầu ra "
                        "sang .avi nếu .mp4 không được hỗ trợ trên máy này."
                    )

            writer.write(out_frame)

            if idx % 10 == 0 or (total and idx == total):
                print(f"[run.py] Đã xử lý {idx}/{total or '?'} khung hình")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[run.py] Đã lưu video (KHÔNG có audio): {output_path}")


def check_ffmpeg():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return False, (
            "Không tìm thấy ffmpeg/ffprobe trong PATH.\n"
            "Tải tại https://ffmpeg.org/download.html — Windows: giải nén rồi thêm "
            "thư mục 'bin' vào biến môi trường PATH, sau đó MỞ LẠI cửa sổ dòng lệnh."
        )
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=15)
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 or "blocked" in combined.lower():
            return False, (
                "ffmpeg.exe bị chặn bởi chính sách bảo mật của máy/tổ chức.\n"
                "Vui lòng liên hệ IT để được whitelist file ffmpeg.exe.\n"
                f"Chi tiết: {combined.strip()[:300]}"
            )
    except Exception as e:
        return False, f"Không chạy được ffmpeg.exe: {e}"
    return True, ""


def get_video_fps(video_path):
    cmd = ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate", video_path]
    out = subprocess.check_output(cmd).decode().strip()
    if "/" in out:
        num, den = out.split("/")
        den = float(den)
        return float(num) / den if den else float(num)
    return float(out)


def has_audio_stream(video_path):
    cmd = ["ffprobe", "-v", "0", "-select_streams", "a", "-show_entries", "stream=index",
           "-of", "csv=p=0", video_path]
    out = subprocess.check_output(cmd).decode().strip()
    return len(out) > 0


def extract_frames(video_path, frames_dir, frame_format="png"):
    os.makedirs(frames_dir, exist_ok=True)
    pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")
    cmd = ["ffmpeg", "-y", "-i", video_path]
    if frame_format == "jpg":
        cmd += ["-qscale:v", "2"]
    cmd += [pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def assemble_video(frames_dir, frame_format, fps, src_video_path, output_video_path,
                    crf=14, preset="slow", has_audio=True):
    pattern = os.path.join(frames_dir, f"frame_%06d.{frame_format}")
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", pattern]
    if has_audio:
        cmd += ["-i", src_video_path]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if has_audio:
        cmd += ["-c:a", "copy"]
    cmd += [output_video_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def resolve_video_output_path(output_path, input_video_path):
    ext = os.path.splitext(output_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        return output_path
    os.makedirs(output_path, exist_ok=True)
    name = os.path.splitext(os.path.basename(input_video_path))[0]
    return os.path.join(output_path, f"{name}_restored.mp4")


def process_video_ffmpeg(video_path, output_path, models, args):
    print(f"[run.py] Video đầu vào: {video_path}")
    fps = get_video_fps(video_path)
    audio = has_audio_stream(video_path)
    print(f"[run.py] FPS: {fps:.3f} | Có audio: {audio}")

    out_video_path = resolve_video_output_path(output_path, video_path)
    work_dir = tempfile.mkdtemp(prefix="codeformer_video_")
    in_frames_dir = os.path.join(work_dir, "in_frames")
    out_frames_dir = os.path.join(work_dir, "out_frames")
    os.makedirs(out_frames_dir, exist_ok=True)

    try:
        print("[run.py] Đang tách khung hình...")
        extract_frames(video_path, in_frames_dir, frame_format=args.frame_format)

        frame_files = sorted(glob.glob(os.path.join(in_frames_dir, f"*.{args.frame_format}")))
        total = len(frame_files)
        if total == 0:
            raise RuntimeError("Không tách được khung hình nào từ video.")
        print(f"[run.py] Tổng số khung hình: {total}")

        for idx, frame_path in enumerate(frame_files, start=1):
            frame_name = os.path.basename(frame_path)
            out_frame_path = os.path.join(out_frames_dir, frame_name)
            try:
                hub.process_image(
                    "restore", models, frame_path, out_frame_path,
                    fidelity_weight=args.fidelity_weight,
                    has_aligned=True,
                    only_center_face=args.only_center_face,
                    draw_box=args.draw_box,
                )
            except Exception as e:
                print(f"  Lỗi khung hình {frame_name}: {e} -> dùng khung gốc")
                shutil.copy(frame_path, out_frame_path)

            if idx % 10 == 0 or idx == total:
                print(f"[run.py] Đã xử lý {idx}/{total} khung hình")

        print("[run.py] Đang ghép lại video...")
        assemble_video(out_frames_dir, args.frame_format, fps, video_path, out_video_path,
                        crf=args.crf, preset=args.video_preset, has_audio=audio)
        print(f"[run.py] Đã lưu video: {out_video_path}")

    finally:
        if not args.keep_frames:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"[run.py] Giữ lại khung hình tạm tại: {work_dir}")


def process_video(video_path, output_path, models, args):
    if args.no_ffmpeg:
        process_video_opencv(video_path, output_path, models, args)
        return
    ok, reason = check_ffmpeg()
    if not ok:
        print(f"[run.py] Không dùng được ffmpeg: {reason}")
        print("[run.py] Tự chuyển sang đường OpenCV (video kết quả sẽ KHÔNG có audio).")
        process_video_opencv(video_path, output_path, models, args)
        return
    process_video_ffmpeg(video_path, output_path, models, args)


# ============================================================
#                          CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeFormer - Restore / Colorize / Inpaint khuôn mặt cho ảnh hoặc video"
    )
    parser.add_argument("-i", "--input_path", type=str, default=None,
                        help="File/thư mục ảnh, hoặc file video (bỏ trống -> hộp thoại chọn file)")
    parser.add_argument("-o", "--output_path", type=str, default="results",
                        help="Thư mục/đường dẫn lưu kết quả")
    parser.add_argument("--task", type=str, default=None, choices=["restore", "colorize", "inpaint"],
                        help="Chọn thẳng tác vụ từ dòng lệnh (bỏ trống -> hiện menu hỏi mỗi ảnh)")

    # ---- Riêng cho restore ----
    parser.add_argument("-w", "--fidelity_weight", type=float, default=0.8,
                        help="[restore] 0-1, nhỏ = đẹp/mượt nhưng dễ bịa mặt, lớn = giữ đặc điểm gốc")
    parser.add_argument("--bg_upsampler", dest="bg_upsampler", action="store_true", default=True,
                        help="[restore] Bật nâng cấp nền bằng Real-ESRGAN (mặc định: bật)")
    parser.add_argument("--no_bg_upsampler", dest="bg_upsampler", action="store_false")
    parser.add_argument("--face_upsample", dest="face_upsample", action="store_true", default=True)
    parser.add_argument("--no_face_upsample", dest="face_upsample", action="store_false")
    parser.add_argument("--upscale", type=int, default=2)
    parser.add_argument("--bg_model", type=str, default="x4plus", choices=["x2plus", "x4plus"])
    parser.add_argument("--bg_tile", type=int, default=0)
    parser.add_argument("--bg_tile_pad", type=int, default=32)
    parser.add_argument("--bg_pre_pad", type=int, default=10)
    parser.add_argument("--bg_blend", type=float, default=1.0)
    parser.add_argument("--bg_sharpen", type=float, default=0.0)
    parser.add_argument("--final_sharpen", type=float, default=0.0)
    parser.add_argument("--face_pre_sharpen", type=float, default=0.0)
    parser.add_argument("--face_post_sharpen", type=float, default=0.0)

    # ---- Chung cho cả 3 tác vụ ----
    parser.add_argument("--has_aligned", action="store_true",
                        help="Input đã là ảnh mặt crop/align 512x512 (chỉ áp dụng cho ảnh). "
                             "BẮT BUỘC nên bật khi dùng --task inpaint để mask trắng chính xác.")
    parser.add_argument("--only_center_face", action="store_true")
    parser.add_argument("--draw_box", action="store_true")
    parser.add_argument("--half", dest="half", action="store_true", default=False)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--crop_ratio", type=float, nargs=2, default=(1.0, 1.0), metavar=("H_RATIO", "W_RATIO"))

    # ---- Riêng cho VIDEO (chỉ áp dụng khi --task restore) ----
    parser.add_argument("--frame_format", type=str, default="png", choices=["png", "jpg"])
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--video_preset", type=str, default="slow",
                        choices=["ultrafast", "fast", "medium", "slow", "veryslow"])
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--no_ffmpeg", action="store_true")

    return parser.parse_args()


def load_models_for_task(task, args, device):
    if task == "restore":
        return hub.load_models(
            "restore", device=device, upscale=args.upscale, bg_upsampler=args.bg_upsampler,
            face_upsample=args.face_upsample, bg_model_name=args.bg_model, bg_tile=args.bg_tile,
            bg_tile_pad=args.bg_tile_pad, bg_pre_pad=args.bg_pre_pad, half=args.half,
            crop_ratio=tuple(args.crop_ratio), bg_blend=args.bg_blend, bg_sharpen=args.bg_sharpen,
            final_sharpen=args.final_sharpen, face_pre_sharpen=args.face_pre_sharpen,
            face_post_sharpen=args.face_post_sharpen,
        )
    # colorize / inpaint: dùng upscale=1 vì 2 model này không phóng to, chỉ sửa tại chỗ
    return hub.load_models(task, device=device, upscale=1, crop_ratio=tuple(args.crop_ratio))


def run_task_on_images(task, img_paths, output_root, models, args):
    for img_path in img_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"[run.py] ({hub.TASK_LABELS[task]}) Xử lý: {img_name}")

        if args.has_aligned:
            out_target = output_root
        else:
            out_target = os.path.join(output_root, f"{img_name}_{task}.png")

        try:
            if task == "restore":
                saved_paths = hub.process_image(
                    "restore", models, img_path, out_target,
                    fidelity_weight=args.fidelity_weight, has_aligned=args.has_aligned,
                    only_center_face=args.only_center_face, draw_box=args.draw_box,
                )
            else:
                saved_paths = hub.process_image(
                    task, models, img_path, out_target,
                    has_aligned=args.has_aligned, only_center_face=args.only_center_face,
                    draw_box=args.draw_box,
                )
            for p in saved_paths:
                print(f"  -> Đã lưu: {p}")
        except Exception as e:
            print(f"  Lỗi xử lý {img_name}: {e}")


def main():
    args = parse_args()

    input_path = args.input_path
    if input_path is None:
        print("[run.py] Vui lòng chọn file ảnh hoặc video...")
        input_path = select_input_path()
        if input_path is None:
            print("[run.py] Không chọn file. Thoát chương trình.")
            return
    print(f"[run.py] Đầu vào: {input_path}")

    video_mode = is_video_file(input_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============ VIDEO: chỉ hỗ trợ tác vụ restore ============
    if video_mode:
        task = args.task or "restore"
        if task != "restore":
            print(f"[run.py] Tác vụ '{task}' hiện chưa hỗ trợ VIDEO (chỉ hỗ trợ ảnh). "
                  f"Tự chuyển sang tác vụ 'restore' cho video này.")
            task = "restore"
        print(f"[run.py] Chế độ: VIDEO | Tác vụ: {hub.TASK_LABELS[task]}")
        models = load_models_for_task(task, args, device)
        try:
            process_video(input_path, args.output_path, models, args)
        except Exception as e:
            print(f"[run.py] Lỗi xử lý video: {e}")
        print("[run.py] Hoàn tất.")
        return

    # ============ ẢNH ============
    img_paths = collect_image_paths(input_path)
    if not img_paths:
        print(f"[run.py] Không tìm thấy ảnh (hoặc video) hợp lệ trong: {input_path}")
        return
    os.makedirs(args.output_path, exist_ok=True)
    print(f"[run.py] Đầu ra: {args.output_path}")
    print(f"[run.py] Chế độ: ẢNH ({len(img_paths)} ảnh)")

    # ---- Nếu chỉ định --task từ dòng lệnh: chạy 1 lần, không hỏi menu ----
    if args.task is not None:
        models = load_models_for_task(args.task, args, device)
        run_task_on_images(args.task, img_paths, args.output_path, models, args)
        print("[run.py] Hoàn tất.")
        return

    # ---- Không chỉ định --task: hỏi menu, có thể lặp lại nhiều tác vụ ----
    while True:
        task = select_task_interactive()
        if task is None:
            print("[run.py] Kết thúc chương trình.")
            return
        models = load_models_for_task(task, args, device)
        run_task_on_images(task, img_paths, args.output_path, models, args)
        # giải phóng model trước khi có thể tải model của tác vụ khác
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()