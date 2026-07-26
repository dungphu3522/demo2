"""
main.py — TRUNG TÂM điều phối.

File này không tự chứa logic xử lý ảnh nữa (để tránh phình to và khó
bảo trì). Nó chỉ:
    1. Định nghĩa danh sách các tác vụ có sẵn (TASKS)
    2. Cung cấp load_models(task, ...) và process_image(task, ...) làm
       cổng vào DUY NHẤT mà run.py cần gọi, bất kể là tác vụ nào.

3 tác vụ thật sự có (khớp với 3 file .pth trong Release chính thức):
    - "restore"  -> task_restore.py   (codeformer.pth)               Làm nét mặt
    - "colorize" -> task_colorize.py  (codeformer_colorization.pth)  Tô màu mặt
    - "inpaint"  -> task_inpaint.py   (codeformer_inpainting.pth)    Vá mặt

Ghi chú: "Enhancing Old Photos / Fixing AI-arts" trên trang demo gốc
CHỈ LÀ CÂU BANNER quảng cáo cho tính năng "restore" ở trên, KHÔNG PHẢI
một model/tác vụ thứ 4 riêng biệt, nên không có file task riêng cho nó.
"""

import task_restore
import task_colorize
import task_inpaint

TASKS = {
    "restore": task_restore,
    "colorize": task_colorize,
    "inpaint": task_inpaint,
}

TASK_LABELS = {
    "restore": "Face Restoration (làm nét khuôn mặt)",
    "colorize": "Face Color Enhancement and Restoration (tô màu mặt đen trắng/phai màu)",
    "inpaint": "Face Inpainting (vá khuôn mặt bị rách/hỏng)",
}


def _get_task_module(task):
    if task not in TASKS:
        raise ValueError(f"task không hợp lệ: '{task}'. Chỉ nhận: {list(TASKS)}")
    return TASKS[task]


def load_models(task, **kwargs):
    """Tải model cho ĐÚNG 1 tác vụ. kwargs được chuyển thẳng cho
    task_xxx.load_models(...) tương ứng (mỗi tác vụ nhận tham số khác nhau,
    xem trong từng file task_*.py)."""
    return _get_task_module(task).load_models(**kwargs)


def process_image(task, models, img_path, output_path, **kwargs):
    """Xử lý 1 ảnh theo ĐÚNG tác vụ mà `models` đã được load cho.
    kwargs được chuyển thẳng cho task_xxx.process_image(...)."""
    return _get_task_module(task).process_image(models, img_path, output_path, **kwargs)