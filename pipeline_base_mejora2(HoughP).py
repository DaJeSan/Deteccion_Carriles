import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================================================================
# RUTAS
# ================================================================
VIDEO_PATH       = os.path.join("imagenes", "video_carretera.mp4")
OUTPUT_VIDEO_PATH = "resultado_mejora2_houghP.avi"
DIAGRAMA_PATH    = "pipeline_mejora2_steps.png"

# ================================================================
# 1. CARGA DE VIDEO
# ================================================================
def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")
    return cap

# ================================================================
# 2. ESCALA DE GRISES
# ================================================================
def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ================================================================
# 3. FILTRO GAUSSIANO
# ================================================================
def gaussian_blur(gray, kernel_size=(9, 9)):
    return cv2.GaussianBlur(gray, kernel_size, sigmaX=0)

# ================================================================
# 4. DETECTOR DE BORDES CANNY 
# ================================================================
def canny_edge_detection(blurred, low_threshold=50, high_threshold=150):
    return cv2.Canny(blurred, low_threshold, high_threshold)

# ================================================================
# 5. FILTRO DE COLOR HSV 
# ================================================================
def hsv_color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([ 18,  40, 100], dtype=np.uint8)
    upper_yellow = np.array([ 30, 255, 255], dtype=np.uint8)
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white  = np.array([  0,   0, 170], dtype=np.uint8)
    upper_white  = np.array([179,  40, 255], dtype=np.uint8)
    mask_white   = cv2.inRange(hsv, lower_white, upper_white)

    mask_combined  = cv2.bitwise_or(mask_yellow, mask_white)
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    return filtered_image, mask_combined, mask_white, mask_yellow

# ================================================================
# 6. REGIÓN DE INTERÉS
# ================================================================
def region_of_interest(edges, image_shape):
    height, width = image_shape[0], image_shape[1]
    bottom_left  = (int(0.15 * width),  int(0.85 * height))
    bottom_right = (int(0.85 * width),  int(0.85 * height))
    top_left     = (int(0.425 * width), int(0.55 * height))
    top_right    = (int(0.575 * width), int(0.55 * height))
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, vertices, mask

# ================================================================
# 7. HOUGH PROBABILÍSTICA CALIBRADA  ← MEJORA 2
# ================================================================
def hough_lines_probabilistic(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        min_line_length=40,
        max_line_gap=80):
    
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    return lines  # shape (N, 1, 4) o None

# ================================================================
# 8. DIBUJAR SEGMENTOS (formato HoughLinesP)
# ================================================================
def draw_lines_probabilistic(image, lines, roi_mask, color=(0, 255, 0), thickness=4):
   
    result = image.copy()
    if lines is None or len(lines) == 0:
        return result

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle_deg < 20 or angle_deg > 160:
            continue
        if abs(angle_deg - 90) < 10:
            continue

        cv2.line(result, (x1, y1), (x2, y2), color, thickness)

    return result

# ================================================================
# 9. PIPELINE COMPLETO — MEJORA 2
# ================================================================
def process_frame(frame):
    
    # Paso 1: Filtro HSV — genera imagen donde solo hay píxeles B/A
    filtered_image, mask_combined, mask_white, mask_yellow = hsv_color_filter(frame)

    # Paso 2: Grises de la imagen YA filtrada (sin ruido de color)
    gray    = to_grayscale(filtered_image)

    # Paso 3: Suavizado
    blurred = gaussian_blur(gray, kernel_size=(9, 9))

    # Paso 4: Canny — solo detecta bordes de píxeles blancos/amarillos
    edges   = canny_edge_detection(blurred, low_threshold=100, high_threshold=300)

    # Paso 5: ROI — recorta al trapecio de interés
    masked_edges, roi_vertices, roi_mask = region_of_interest(edges, frame.shape)

    filtered_edges = masked_edges

    # Paso 6: HoughLinesP  ← MEJORA 2
    lines = hough_lines_probabilistic(
        masked_edges,          
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        min_line_length=40,
        max_line_gap=80
    )

    # Paso 8: Dibujo
    result = draw_lines_probabilistic(frame, lines, roi_mask, color=(0, 255, 0), thickness=4)
    cv2.polylines(result, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=2)

    debug_images = {
        "filtered_bgr":   filtered_image,
        "mask_white":     mask_white,
        "mask_yellow":    mask_yellow,
        "mask_combined":  mask_combined,
        "grayscale":      gray,
        "gaussian":       blurred,
        "canny":          edges,
        "filtered_edges": filtered_edges,
        "roi_mask":       roi_mask,
        "roi_vertices":   roi_vertices,
    }
    return result, debug_images

# ================================================================
# 10. DIAGRAMA DEL PIPELINE
# ================================================================
def show_pipeline_diagram(frame, debug_images):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        "Pipeline Mejora 2 — HoughLinesP calibrada",
        fontsize=11, fontweight='bold'
    )

    frame_rgb    = cv2.cvtColor(frame,                       cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(debug_images["filtered_bgr"], cv2.COLOR_BGR2RGB)
    result_rgb   = cv2.cvtColor(debug_images.get("result_bgr", frame), cv2.COLOR_BGR2RGB)

    titles = [
        "1. Original (BGR)",
        "2. Máscara blanco",
        "3. Máscara amarillo",
        "4. Imagen filtrada HSV",
        "5. Grises → Gaussiano",
        "6. Canny 50/150",
        "7. Bordes en ROI (Canny+ROI)",
        "8. Resultado HoughLinesP",
    ]
    images = [
        frame_rgb,
        debug_images["mask_white"],
        debug_images["mask_yellow"],
        filtered_rgb,
        debug_images["gaussian"],
        debug_images["canny"],
        debug_images["filtered_edges"],
        result_rgb,
    ]
    cmaps = [None, 'gray', 'gray', None, 'gray', 'gray', 'gray', None]

    for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(DIAGRAMA_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Diagrama guardado en '{DIAGRAMA_PATH}'")

# ================================================================
# 11. PROCESAR VIDEO COMPLETO
# ================================================================
def process_video(video_path, show_preview=True, save_video=True):
    cap    = open_video(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"=== Video: {video_path} ===")
    print(f"  Resolución  : {width}×{height}")
    print(f"  FPS         : {fps}")
    print(f"  Total frames: {total}")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        print(f"Guardando en '{OUTPUT_VIDEO_PATH}'")

    frame_count = 0
    first_done  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result, debug_images = process_frame(frame)
        debug_images["result_bgr"] = result

        if not first_done:
            show_pipeline_diagram(frame, debug_images)
            first_done = True

        if writer is not None:
            writer.write(result)

        if show_preview:
            cv2.imshow("Mejora 2 — HoughLinesP", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Interrumpido en frame {frame_count}")
                break

        if frame_count % 50 == 0 or frame_count == 1:
            print(f"  Frame {frame_count}/{total}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n=== Completado: {frame_count} frames ===")

# ================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MEJORA 2: Hough Probabilística calibrada (HoughLinesP)")
    print("  Flujo: HSV → Gris → Gauss → Canny → ROI → HoughLinesP")
    print("=" * 60)
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encontró el video en '{VIDEO_PATH}'")
        sys.exit(1)
    process_video(video_path=VIDEO_PATH, show_preview=True, save_video=True)