import cv2

def main():
    # Índice de cámara: prueba 1 para Camo Studio

    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print("Cámara encontrada en índice:", i)
            cap.release()

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara con índice {cam_index}")
        return

    print("Cámara abierta correctamente 🎥")
    print("Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener frame")
            break

        cv2.imshow("Camo Studio Camera", frame)

        # salir con q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
