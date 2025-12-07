import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
import time
from collections import defaultdict, deque

import torch
from torchvision import transforms
from ultralytics import YOLO
from timm import create_model

# ============================
# === TU CONFIGURACIÓN ===
# ============================
#name = "drowning_1"
name = "inges"
ultimos_frames = 8

video_path = f"Pruebas/Videos/{name}.MOV"
output_path = f"Pruebas/Resultados/resultado_{name}_god.avi"

yolo_model_path = "Modelos/YOLO/best.pt"
pit_model_path = "Modelos/PIT/pit_best_model.pth"

bytetrack_path = "Modelos/YOLO/bytetrack_custom.yaml"

# Tamaños de visualización (ajustables)
DISPLAY_WIDTH = 1200   # ancho total de la ventana con los 2 videos
DISPLAY_HEIGHT = 540   # altura de la ventana (ambos videos compartirán esta altura)
LEFT_WIDTH = DISPLAY_WIDTH // 2
RIGHT_WIDTH = DISPLAY_WIDTH - LEFT_WIDTH
STOP_EVENT = threading.Event()
VIDEO_FPS = 30

# Queues y control
FRAME_QUEUE = queue.Queue(maxsize=6)
DISPLAY_QUEUE = queue.Queue(maxsize=6)
STOP_EVENT = threading.Event()

# Clasificación cada N frames por ID (puedes ajustar)
CLASSIFY_EVERY_N = 4


# ============================
# === UTIL: carga modelos ===
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    # YOLO (usa tu ruta)
    yolo = YOLO(yolo_model_path)

    # PiT
    pit = create_model('pit_b_224', pretrained=False, num_classes=2)
    pit.load_state_dict(torch.load(pit_model_path, map_location=device))
    pit.to(device)
    pit.eval()

    # Transform (igual que entrenaste)
    pit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return yolo, pit, pit_transform


# ============================
# === THREAD: captura video ==
# ============================
def capture_frames():
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or video_fps is None:
        video_fps = 30
    global VIDEO_FPS
    VIDEO_FPS = video_fps
    print(f"FPS detectado: {video_fps}")


    if not cap.isOpened():
        print("ERROR: no se pudo abrir video:", video_path)
        STOP_EVENT.set()
        return

    while not STOP_EVENT.is_set():
        ret, frame = cap.read()

        if not ret:
            # fin de video
            break

        # Si la cola está llena, descartamos el frame más viejo para no acumular lag
        if FRAME_QUEUE.full():
            try:
                FRAME_QUEUE.get_nowait()
            except queue.Empty:
                pass

        FRAME_QUEUE.put(frame)

        # pequeño sleep opcional para no agotar CPU (ajusta si quieres)
        time.sleep(0.001)

    cap.release()
    # señal para detener el procesamiento si ya no hay más frames
    FRAME_QUEUE.put(None)


# ============================
# === Clasificación PiT =====
# ============================
from PIL import Image as PILImage

def classify_roi_with_pit(pit, transform, roi_bgr):
    # ROI viene en BGR (OpenCV). Convertir a PIL RGB
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
    # Mantener buffers por track_id
    buffers = defaultdict(lambda: deque(maxlen=ultimos_frames))
    id_frame_counter = defaultdict(int)

    # Para guardar video con overlays si quieres (opcional)
    # Obtendremos dimensiones del primer frame procesado al vuelo para el writer
    video_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #out = cv2.VideoWriter(output_path, fourcc, video_fps, (frame_width, frame_height))


    while True:
        frame = FRAME_QUEUE.get()
        if frame is None:
            # fin de captura
            break

        original = frame.copy()

        # Ejecutar tracking/detección
        try:
            # Usamos el mismo call que tenías para asegurar compatibilidad con bytetrack
            results = yolo.track(frame, persist=True, conf=0.25, tracker=bytetrack_path)[0]
        except Exception as e:
            # En caso de error en YOLO, dejamos el frame sin procesar
            print("WARN: YOLO track fallo:", e)
            processed = frame
            # Encolar par (original, processed)
            if DISPLAY_QUEUE.full():
                try:
                    DISPLAY_QUEUE.get_nowait()
                except queue.Empty:
                    pass
            DISPLAY_QUEUE.put((original, processed))
            continue

        processed = frame.copy()

        # Si no hay cajas, empujar ambos frames iguales
        if not hasattr(results, "boxes") or len(results.boxes) == 0:
            if DISPLAY_QUEUE.full():
                try:
                    DISPLAY_QUEUE.get_nowait()
                except queue.Empty:
                    pass
            DISPLAY_QUEUE.put((original, processed))
            continue

        # Recorrer detecciones
        for box in results.boxes:
            # box.xyxy, box.id, ...
            try:
                xy = box.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2]
            except:
                xy = box.xyxy[0].numpy()
            x1, y1, x2, y2 = map(int, xy)

            track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
            # Asegurar contadores y buffers
            if track_id != -1:
                id_frame_counter[track_id] += 1
                # crear buffer si no existe (defaultdict ya hace)
            else:
                # para detecciones sin ID, asignar -1 y tratar como "swimming" por defecto
                pass

            # Clasificar solo cada N frames por ese ID
            cls_pred = None
            conf = 0.0
            if track_id != -1:
                if id_frame_counter[track_id] % CLASSIFY_EVERY_N == 0:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size != 0:
                        cls_pred, conf = classify_roi_with_pit(pit, pit_transform, roi)
                        # si clasificación OK, push al buffer
                        if cls_pred is not None:
                            buffers[track_id].append(cls_pred)
                else:
                    # usar la clase más frecuente en el buffer si existe
                    if len(buffers[track_id]) > 0:
                        cls_pred = max(set(buffers[track_id]), key=buffers[track_id].count)
                        conf = 1.0
                    else:
                        cls_pred = 1  # default swimming
                        conf = 0.0
            else:
                cls_pred = 1
                conf = 0.0

            # Dibujar bounding box + label
            color = (0, 0, 255) if cls_pred == 0 else (255, 0, 0)
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            text = f"ID {track_id} - {('AHOG' if cls_pred==0 else 'NAD')}: {conf:.2f}"
            cv2.putText(processed, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # opcional: escribir video de salida si quieres (aquí está preparado)
        if video_writer is None:
            h, w = processed.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (w, h))

        if video_writer is not None:
            video_writer.write(processed)

        # Encolar (original, processed)
        if DISPLAY_QUEUE.full():
            try:
                DISPLAY_QUEUE.get_nowait()
            except queue.Empty:
                pass
        DISPLAY_QUEUE.put((original, processed))

    # cerrar writer si fue creado
    if 'video_writer' in locals() and video_writer is not None:
        video_writer.release()


# ============================
# === GUI: mostrar lado a lado =
# ============================
def start_gui():
    root = tk.Tk()
    root.title("Original (izq)  |  Detección (der) — Tiempo Real")

    # Forzar tamaño de ventana
    root.geometry(f"{DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    label = tk.Label(root)
    label.pack(fill=tk.BOTH, expand=True)

    # función interna que actualiza la imagen periódicamente
    def gui_update():
        if STOP_EVENT.is_set() and DISPLAY_QUEUE.empty():
            root.quit()
            return

        try:
            original, processed = DISPLAY_QUEUE.get_nowait()
        except queue.Empty:
            # si no hay frame, reintentar pronto
            label.after(10, gui_update)
            return

        # Resize original y processed para que sean del mismo alto (DISPLAY_HEIGHT)
        # Calculamos tamaño proporcional manteniendo aspect ratio
        def resize_to_height(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, target_h))

        # queremos que ambos tengan misma altura = DISPLAY_HEIGHT
        target_h = DISPLAY_HEIGHT
        orig_r = resize_to_height(original, target_h)
        proc_r = resize_to_height(processed, target_h)

        # si el ancho combinado es mayor a DISPLAY_WIDTH, redimensionamos ambos proporcionalmente
        combined_w = orig_r.shape[1] + proc_r.shape[1]
        if combined_w > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / combined_w
            new_h = int(target_h * scale)
            orig_r = cv2.resize(orig_r, (int(orig_r.shape[1]*scale), new_h))
            proc_r = cv2.resize(proc_r, (int(proc_r.shape[1]*scale), new_h))

        # Alineamos horizontalmente (hconcat requiere misma altura)
        try:
            combined = cv2.hconcat([orig_r, proc_r])
        except Exception:
            # fallback: mostrar solo processed si concat falla
            combined = proc_r

        # Convertir BGR->RGB y a PIL ImageTk
        img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        im_pil = PILImage.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)

        label.imgtk = imgtk
        label.configure(image=imgtk)

        # programar siguiente update
        label.after(10, gui_update)

    # Iniciar ciclo
    label.after(10, gui_update)
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def on_close():
    # marcar stop y cerrar
    STOP_EVENT.set()


# ============================
# === MAIN: orquesta todo ===
# ============================
def main():
    yolo, pit, pit_transform = load_models()
    # Hilos
    t_cap = threading.Thread(target=capture_frames, daemon=True)
    t_proc = threading.Thread(target=processing_thread, args=(yolo, pit, pit_transform), daemon=True)

    t_cap.start()
    t_proc.start()

    # Ejecutar GUI (bloqueante) en hilo principal
    start_gui()

    # cuando GUI termine, esperar hilos
    STOP_EVENT.set()
    t_cap.join(timeout=1.0)
    t_proc.join(timeout=1.0)
    print("FIN.")


if __name__ == "__main__":
    main()
