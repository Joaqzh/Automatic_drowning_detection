import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

from torchvision import transforms, models
import torch
import torch.nn as nn
from collections import deque
from timm import create_model
import os


# === Configura Prueba ===
name = "drowning_1"
ultimos_frames = 8

video_path = f"Pruebas/Videos/{name}.MOV"
output_path = f"Pruebas/Resultados/resultado_{name}.avi"

yolo_model_path = "Modelos/YOLO/best.pt"
pit_model_path = "Modelos/PIT/pit_best_model.pth"

bytetrack_path = "Modelos/YOLO/bytetrack_custom.yaml"

is_stream = video_path.startswith("rtsp") or video_path.isnumeric() or video_path == "0"

# Cargar modelo PIT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pit = create_model('pit_b_224', pretrained=False, num_classes=2)
pit.load_state_dict(torch.load(pit_model_path, map_location=device))
pit.to(device)
pit.eval()

# Transformaciones Pit
pit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Cargar modelo YOLO
model = YOLO(yolo_model_path)
class_names = model.names  # Diccionario de clases

# Colores por clase
class_colors = {
    0: (0, 0, 255),    # Rojo - ahogándose
    1: (255, 0, 0)     # Azul - nadando
}

# Abrir video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Configurar escritor de video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

# Crear ventana Tkinter
root = tk.Tk()
root.title("YOLOv8 + PiT - Detección con Tracking")

# Dimensiones de la ventana
display_width = 800
display_height = 600
canvas = tk.Canvas(root, width=display_width, height=display_height)
canvas.pack()

recent_classes = deque(maxlen = ultimos_frames)
alert_active = False

def classify_with_pit(roi):
    pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    input_tensor = pit_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = pit(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = torch.argmax(probs).item()
        conf = float(probs[pred])

    return pred, conf

def draw_boxes(frame, results):
    frame_class_preds = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # === CLASIFICACIÓN PiT ===
        cls_id, conf = classify_with_pit(roi)
        frame_class_preds.append(cls_id)

        # Track ID
        track_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else -1
        label = f"ID {track_id} - {class_names[cls_id]}: {conf:.2f}"

        color = class_colors.get(cls_id, (0, 255, 0))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, frame_class_preds

def update_frame():
    global alert_active

    ret, frame = cap.read()
    if ret:
        frame_classes = []

        results = model.track(frame, persist=True, conf=0.25, tracker=bytetrack_path)[0]
        annotated_frame, frame_classes = draw_boxes(frame.copy(), results)

        # Contar si hay mayoría de "drowning" en los ultimos frames
        if 0 in frame_classes:
            recent_classes.append(0)
        else:
            recent_classes.append(1)

        if recent_classes.count(0) > len(recent_classes) // 2:
            if not alert_active:
                print("🚨 ALERTA: Posible ahogamiento detectado.")
                alert_active = True
        else:
            if alert_active:
                print("✅ ALERTA DESACTIVADA")
                alert_active = False

        # === Mostrar alerta en pantalla si está activa ===
        if alert_active:
            cv2.putText(
                annotated_frame,
                "ALERTA: POSIBLE AHOGAMIENTO",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # Rojo
                3
            )

        # Mostrar en Tkinter
        img_display = cv2.resize(annotated_frame, (display_width, display_height))
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_display))

        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img = img_tk

        # Guardar frame al archivo de salida
        out.write(annotated_frame)

        # Continuar con el siguiente frame
        root.after(1, update_frame)
    else:
        cap.release()
        out.release()
        print("✅ Video guardado en:", output_path)
        root.destroy()

# Iniciar interfaz
update_frame()
root.mainloop()  