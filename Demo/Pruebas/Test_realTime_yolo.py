import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
import time

from ultralytics import YOLO

# ============================
# === CONFIGURACIÓN ==========
# ============================
name = "inges"
video_path = f"Pruebas/Videos/{name}.MOV"
output_path = f"Pruebas/Resultados/resultado_{name}_yolo.avi"

yolo_model_path = "Modelos/YOLO/best.pt"
bytetrack_path = "Modelos/YOLO/bytetrack_custom.yaml"

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 540
STOP_EVENT = threading.Event()
VIDEO_FPS = 30

FRAME_QUEUE = queue.Queue(maxsize=6)
DISPLAY_QUEUE = queue.Queue(maxsize=6)


# ============================
# === CARGAR YOLO ============
# ============================
def load_model():
    yolo = YOLO(yolo_model_path)
    return yolo


# ============================
# === THREAD: CAPTURA VIDEO ==
# ============================
def capture_frames():
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    global VIDEO_FPS
    VIDEO_FPS = fps if fps > 0 else 30

    if not cap.isOpened():
        print("ERROR al abrir video.")
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
# === THREAD: PROCESAMIENTO ==
# ============================
def processing_thread(yolo):
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    while True:
        frame = FRAME_QUEUE.get()
        if frame is None:
            break

        original = frame.copy()

        # YOLO detección + tracking
        try:
            results = yolo.track(frame, persist=True, conf=0.25, tracker=bytetrack_path)[0]
        except Exception as e:
            print("YOLO ERROR:", e)
            processed = frame
            enqueue_display(original, processed)
            continue

        processed = frame.copy()

        if hasattr(results, "boxes"):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                label = results.names[cls_id].upper()

                # Colores por clase
                color = (0, 0, 255) if cls_id == 0 else (255, 0, 0)

                cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    processed,
                    f"ID {track_id} - {label}: {conf:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # Inicializar writer
        if video_writer is None:
            h, w = processed.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w, h))

        video_writer.write(processed)
        enqueue_display(original, processed)

    if video_writer is not None:
        video_writer.release()


def enqueue_display(original, processed):
    if DISPLAY_QUEUE.full():
        try:
            DISPLAY_QUEUE.get_nowait()
        except:
            pass
    DISPLAY_QUEUE.put((original, processed))


# ============================
# === GUI ====================
# ============================
def start_gui():
    root = tk.Tk()
    root.title("Original | YOLO")
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

        def resize_h(img, h):
            ih, iw = img.shape[:2]
            scale = h / ih
            return cv2.resize(img, (int(iw * scale), h))

        orig_r = resize_h(original, DISPLAY_HEIGHT)
        proc_r = resize_h(processed, DISPLAY_HEIGHT)

        total_w = orig_r.shape[1] + proc_r.shape[1]
        if total_w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / total_w
            new_h = int(DISPLAY_HEIGHT * scale)
            orig_r = cv2.resize(orig_r, (int(orig_r.shape[1] * scale), new_h))
            proc_r = cv2.resize(proc_r, (int(proc_r.shape[1] * scale), new_h))

        try:
            combined = cv2.hconcat([orig_r, proc_r])
        except:
            combined = proc_r

        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

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
    yolo = load_model()

    t1 = threading.Thread(target=capture_frames, daemon=True)
    t2 = threading.Thread(target=processing_thread, args=(yolo,), daemon=True)

    t1.start()
    t2.start()

    start_gui()

    STOP_EVENT.set()
    t1.join(timeout=1)
    t2.join(timeout=1)
    print("FIN.")


if __name__ == "__main__":
    main()
