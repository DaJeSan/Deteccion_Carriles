import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
# ================================================================
# RUTAS
# ================================================================
VIDEO_PATH = os.path.join("imagenes", "video_carretera.mp4")
OUTPUT_VIDEO_PATH = "resultado_video.avi"
DIAGRAMA_PATH = "pipeline_steps.png"
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
# ================================================================
# 3. FILTRO GAUSSIANO (Suavizado)
# ================================================================
def gaussian_blur(gray, kernel_size=(9, 9)):
    blurred = cv2.GaussianBlur(gray, kernel_size, sigmaX=0)
    return blurred
# ================================================================
# 4. DETECTOR DE BORDES CANNY
# ================================================================
def canny_edge_detection(blurred, low_threshold, high_threshold):
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges
# ================================================================
# 5. REGIÓN DE INTERÉS (ROI) — MÁSCARA TRAPEZOIDAL
# ================================================================
def region_of_interest(edges, image_shape):
    height, width = image_shape[0], image_shape[1]
    bottom_left  = (int(0.15 * width), int(0.85 * height))
    bottom_right = (int(0.85 * width), int(0.85 * height))
    top_left     = (int(0.425 * width), int(0.55 * height))
    top_right    = (int(0.575 * width), int(0.55 * height))
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]],
                        dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, vertices, mask
# ================================================================
# 6. TRANSFORMADA DE HOUGH (ESTÁNDAR)
# ================================================================
def hough_lines(masked_edges, rho=1, theta=np.pi / 180, threshold=100):
    lines = cv2.HoughLines(masked_edges,
                            rho=rho,
                            theta=theta,
                            threshold=threshold)
    return lines
# ================================================================
# 7. DIBUJAR LÍNEAS SOBRE LA IMAGEN ORIGINAL
# ================================================================
def draw_lines(image, lines, roi_mask, color=(0, 255, 0), thickness=4):
    result = image.copy()
    if lines is not None and len(lines) > 0:
        line_canvas = np.zeros_like(image)
        for line in lines:
            rho, theta = line[0]
            angle_deg = abs(np.degrees(theta))
            if angle_deg < 20 or angle_deg > 160:
                continue
            if abs(angle_deg - 90) < 10:
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 2000 * (-b))
            y1 = int(y0 + 2000 * (a))
            x2 = int(x0 - 2000 * (-b))
            y2 = int(y0 - 2000 * (a))
            cv2.line(line_canvas, (x1, y1), (x2, y2), color, thickness)
        roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        line_canvas = cv2.bitwise_and(line_canvas, roi_mask_3ch)
        cv2.add(result, line_canvas, result)
    return result
# ================================================================
# 8. PIPELINE PARA UN SOLO FRAME
# ================================================================
def process_frame(frame):
    gray = to_grayscale(frame)
    blurred = gaussian_blur(gray, kernel_size=(9, 9))
    edges = canny_edge_detection(blurred, low_threshold=150, high_threshold=300)
    masked_edges, roi_vertices, roi_mask = region_of_interest(edges, frame.shape)
    lines = hough_lines(masked_edges,
                        rho=1,
                        theta=np.pi / 180,
                        threshold=100)
    result = draw_lines(frame, lines, roi_mask, color=(0, 255, 0), thickness=4)
    cv2.polylines(result, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=2)
    debug_images = {
        "grayscale": gray,
        "gaussian": blurred,
        "canny": edges,
        "roi_mask": roi_mask,
        "masked_edges": masked_edges,
        "roi_vertices": roi_vertices,
    }
    return result, debug_images
# ================================================================
# 9. MOSTRAR DIAGRAMA DEL PIPELINE (primer frame con matplotlib)
# ================================================================
def show_pipeline_diagram(frame, debug_images):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Pipeline Base de Detección de Carriles",
                 fontsize=16, fontweight='bold')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    titles = [
        "1. Imagen Original",
        "2. Escala de Grises",
        "3. Gaussiano (9×9)",
        "4. Canny (50 / 150)",
        "5. ROI Trapezoidal",
        "6. Resultado (Hough)"
    ]
    images = [
        frame_rgb,
        debug_images["grayscale"],
        debug_images["gaussian"],
        debug_images["canny"],
        debug_images["masked_edges"],
        cv2.cvtColor(debug_images.get("result_bgr", frame), cv2.COLOR_BGR2RGB)
    ]
    cmaps = [None, 'gray', 'gray', 'gray', 'gray', None]
    for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(DIAGRAMA_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Diagrama del pipeline guardado en '{DIAGRAMA_PATH}'")
# ================================================================
# 10. PROCESAR VIDEO COMPLETO
# ================================================================
def process_video(video_path, show_preview=True, save_video=True):
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"=== Video: {video_path} ===")
    print(f"  Resolución  : {width}×{height}")
    print(f"  FPS         : {fps}")
    print(f"  Total frames: {total_frames}")
    print()
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        print(f"Guardando resultado en '{OUTPUT_VIDEO_PATH}'")
        print()
    frame_count = 0
    first_frame_processed = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        result, debug_images = process_frame(frame)
        debug_images["result_bgr"] = result
        if not first_frame_processed:
            print("Generando diagrama del pipeline con el primer frame...")
            show_pipeline_diagram(frame, debug_images)
            first_frame_processed = True
            print()
        if writer is not None:
            writer.write(result)
        if show_preview:
            cv2.imshow("Pipeline Base - Detección de Carriles", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Interrumpido por el usuario en el frame {frame_count}")
                break
        if frame_count % 50 == 0 or frame_count == 1:
            print(f"  Procesado frame {frame_count}/{total_frames}")
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print()
    print(f"=== Procesamiento completado: {frame_count} frames ===")
    if save_video:
        print(f"Video guardado en: {OUTPUT_VIDEO_PATH}")
# ================================================================
# EJECUCIÓN PRINCIPAL
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE BASE: Detección de Carriles")
    print("  Flujo: Imagen -> Gris -> Gaussiano -> Canny -> ROI -> Hough")
    print("=" * 60)
    print()
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encontró el video en '{VIDEO_PATH}'")
        print("Asegúrate de que la carpeta 'imagenes/' contiene 'video_carretera.mp4'")
        sys.exit(1)
    else:
        process_video(
            video_path=VIDEO_PATH,
            show_preview=True,
            save_video=True
        )
