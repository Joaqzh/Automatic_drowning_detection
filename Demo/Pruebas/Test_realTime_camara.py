import cv2
import tkinter as tk
import threading
import queue
import time
from collections import defaultdict, deque
import os

import torch
from torchvision import transforms
from ultralytics import YOLO
from timm import create_model

# Mueve PIL al final
from PIL import Image, ImageTk

# ============================
# === TU CONFIGURACIÓN ===
# ============================
name = "swimming_1"
ultimos_frames = 8

video_path = f"Pruebas/Videos/{name}.MOV"
output_path = f"Pruebas/Resultados/resultado_{name}.avi"

yolo_model_path = "Modelos/YOLO/best.pt"
pit_model_path = "Modelos/PIT/pit_best_model.pth"

bytetrack_path = "Modelos/YOLO/bytetrack_custom.yaml"

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 540

LEFT_WIDTH = DISPLAY_WIDTH // 2
RIGHT_WIDTH = DISPLAY_WIDTH - LEFT_WIDTH
STOP_EVENT = threading.Event()
VIDEO_FPS = 30

FRAME_QUEUE = queue.Queue(maxsize=6)
DISPLAY_QUEUE = queue.Queue(maxsize=6)
STOP_EVENT = threading.Event()

CLASSIFY_EVERY_N = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# === NUEVO: seleccionar video o cámara ===
# ============================
def get_camera_source():
    """
    1. Si existe un video, lo usa.
    2. Si no, busca cámaras (incluye Camo).
    """
    if os.path.exists(video_path):
        print("📹 Usando video:", video_path)
        return video_path

    print("🎥 No se encontró video, buscando cámaras...")

    for cam_index in range(10):
        test = cv2.VideoCapture(cam_index)
        if test.isOpened():
            print(f"✔ Cámara encontrada en índice {cam_index}")
            test.release()
            return cam_index

    print("❌ ERROR: No se encontraron cámaras")
    return None


# ============================
# === UTIL: carga modelos ===
# ============================
def load_models():
    yolo = YOLO(yolo_model_path)

    pit = create_model("pit_b_224", pretrained=False, num_classes=2)
    pit.load_state_dict(torch.load(pit_model_path, map_location=device))
    pit.to(device)
    pit.eval()

    pit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return yolo, pit, pit_transform


# ============================
# === THREAD: captura video/cámara ===
# ============================
def capture_frames():
    source = get_camera_source()
    if source is None:
        STOP_EVENT.set()
        return

    cap = cv2.VideoCapture(source)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or video_fps is None:
        video_fps = 30

    global VIDEO_FPS
    VIDEO_FPS = video_fps
    print(f"FPS detectado: {video_fps}")

    if not cap.isOpened():
        print("ERROR: no se pudo abrir video/cámara:", source)
        STOP_EVENT.set()
        return

    while not STOP_EVENT.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        if FRAME_QUEUE.full():
            try:
                FRAME_QUEUE.get_nowait()
            except queue.Empty:
                pass

        FRAME_QUEUE.put(frame)
        time.sleep(0.001)

    cap.release()
    FRAME_QUEUE.put(None)


# ============================
# === Clasificación PiT =====
# ============================
from PIL import Image as PILImage

def classify_roi_with_pit(pit, transform, roi_bgr):
    try:
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None, 0.0

    pil = PILImage.fromarray(roi_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = pit(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

    return pred, conf


# ============================
# === PROCESSING THREAD ======
# ============================
def processing_thread(yolo, pit, pit_transform):
    buffers = defaultdict(lambda: deque(maxlen=ultimos_frames))
    id_frame_counter = defaultdict(int)

    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    while True:
        frame = FRAME_QUEUE.get()
        if frame is None:
            break

        original = frame.copy()

        try:
            results = yolo.track(frame, persist=True, conf=0.25, tracker=bytetrack_path)[0]
        except Exception as e:
            print("WARN: YOLO track fallo:", e)
            processed = frame

            if DISPLAY_QUEUE.full():
                try: DISPLAY_QUEUE.get_nowait()
                except queue.Empty: pass
            DISPLAY_QUEUE.put((original, processed))
            continue

        processed = frame.copy()

        if not hasattr(results, "boxes") or len(results.boxes) == 0:
            if DISPLAY_QUEUE.full():
                try: DISPLAY_QUEUE.get_nowait()
                except queue.Empty: pass
            DISPLAY_QUEUE.put((original, processed))
            continue

        for box in results.boxes:
            xy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xy)

            track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1

            if track_id != -1:
                id_frame_counter[track_id] += 1

            cls_pred = None
            conf = 0.0

            if track_id != -1:
                if id_frame_counter[track_id] % CLASSIFY_EVERY_N == 0:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        cls_pred, conf = classify_roi_with_pit(pit, pit_transform, roi)

                        if cls_pred is not None:
                            buffers[track_id].append(cls_pred)
                else:
                    if len(buffers[track_id]) > 0:
                        cls_pred = max(set(buffers[track_id]), key=buffers[track_id].count)
                        conf = 1.0
                    else:
                        cls_pred = 1
                        conf = 0.0
            else:
                cls_pred = 1
                conf = 0.0

            color = (0, 0, 255) if cls_pred == 0 else (255, 0, 0)
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            text = f"ID {track_id} - {('AHOG' if cls_pred==0 else 'NAD')}: {conf:.2f}"
            cv2.putText(processed, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if video_writer is None:
            h, w = processed.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w, h))

        if video_writer is not None:
            video_writer.write(processed)

        if DISPLAY_QUEUE.full():
            try: DISPLAY_QUEUE.get_nowait()
            except queue.Empty: pass

        DISPLAY_QUEUE.put((original, processed))

    if video_writer is not None:
        video_writer.release()


# ============================
# === GUI ====================
# ============================
def start_gui():
    root = tk.Tk()
    root.title("Original | Detección — Tiempo Real")
    root.geometry(f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    label = tk.Label(root)
    label.pack(fill=tk.BOTH, expand=True)

    def gui_update():
        if STOP_EVENT.is_set() and DISPLAY_QUEUE.empty():
            root.quit()
            return

        try:
            original, processed = DISPLAY_QUEUE.get_nowait()
        except queue.Empty:
            label.after(10, gui_update)
            return

        def resize_to_height(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h))

        target_h = DISPLAY_HEIGHT
        orig_r = resize_to_height(original, target_h)
        proc_r = resize_to_height(processed, target_h)

        combined_w = orig_r.shape[1] + proc_r.shape[1]
        if combined_w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / combined_w
            new_h = int(target_h * scale)
            orig_r = cv2.resize(orig_r, (int(orig_r.shape[1]*scale), new_h))
            proc_r = cv2.resize(proc_r, (int(proc_r.shape[1]*scale), new_h))

        try:
            combined = cv2.hconcat([orig_r, proc_r])
        except:
            combined = proc_r

        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)

        label.imgtk = imgtk
        label.configure(image=imgtk)
        label.after(10, gui_update)

    label.after(10, gui_update)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def on_close():
    STOP_EVENT.set()


# ============================
# === MAIN ===================
# ============================
def main():
    yolo, pit, pit_transform = load_models()

    t_cap = threading.Thread(target=capture_frames, daemon=True)
    t_proc = threading.Thread(target=processing_thread,
                              args=(yolo, pit, pit_transform),
                              daemon=True)

    t_cap.start()
    t_proc.start()

    start_gui()

    STOP_EVENT.set()
    t_cap.join(timeout=1.0)
    t_proc.join(timeout=1.0)
    print("FIN.")


if __name__ == "__main__":
    main()
