import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# ================================================================
# RUTAS
# ================================================================
VIDEO_PATH        = os.path.join("imagenes", "video_curvas.mp4")
OUTPUT_VIDEO_PATH = "resultado_mejora1_curvas.avi"
DIAGRAMA_PATH     = "pipeline_mejora1_curvas_steps.png"

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
# 3. FILTRO GAUSSIANO  (sin cambios)
# ================================================================
def gaussian_blur(gray, kernel_size=(9, 9)):
    return cv2.GaussianBlur(gray, kernel_size, sigmaX=0)

# ================================================================
# 4. CANNY — umbrales bajados para sombras
# ================================================================
def canny_edge_detection(blurred, low_threshold=30, high_threshold=90):
    return cv2.Canny(blurred, low_threshold, high_threshold)

# ================================================================
# 5. FILTRO HSV — recalibrado para curvas con sombra
# ================================================================
def hsv_color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([ 15,  35,  80], dtype=np.uint8)
    upper_yellow = np.array([ 35, 255, 255], dtype=np.uint8)
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_veg    = cv2.inRange(hsv,
                              np.array([ 35, 50,  50], dtype=np.uint8),
                              np.array([ 85,255, 255], dtype=np.uint8))
    mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_veg))

    lower_white = np.array([  0,   0, 120], dtype=np.uint8)
    upper_white = np.array([179,  35, 255], dtype=np.uint8)
    mask_white  = cv2.inRange(hsv, lower_white, upper_white)

    mask_combined = cv2.bitwise_or(mask_yellow, mask_white)
    kernel        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)
    return filtered_image, mask_combined, mask_white, mask_yellow

# ================================================================
# 6. ROI — trapecio ampliado para curvas
# ================================================================
def region_of_interest(edges, image_shape):
    height, width = image_shape[0], image_shape[1]
    bottom_left  = (int(0.05 * width), int(0.92 * height))
    bottom_right = (int(0.95 * width), int(0.92 * height))
    top_left     = (int(0.30 * width), int(0.60 * height))
    top_right    = (int(0.70 * width), int(0.60 * height))
    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, vertices, mask

# ================================================================
# 7. HOUGH ESTÁNDAR — threshold recalibrado
# ================================================================
def hough_lines(masked_edges, rho=1, theta=np.pi / 180, threshold=80):
    lines = cv2.HoughLines(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold
    )
    return lines

# ================================================================
# 8. DIBUJAR LÍNEAS — selector por lado
# ================================================================
def draw_lines_curvas(image, lines, roi_mask, image_shape):
    result    = image.copy()
    h, w      = image_shape[:2]
    mid       = w // 2
    y_ref     = int(0.80 * h)
    roi_3ch   = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)

    left_drawn  = False
    right_drawn = False

    if lines is None or len(lines) == 0:
        return result

    for line in lines:
        if left_drawn and right_drawn:
            break

        rho_v, theta_v = line[0]
        ang = abs(np.degrees(theta_v))

        if ang < 15 or ang > 165:
            continue
        if 80 < ang < 100:
            continue

        a  = np.cos(theta_v)
        b  = np.sin(theta_v)

        if abs(a) < 1e-6:
            continue
        x0     = a * rho_v
        y0     = b * rho_v
        t      = (y_ref - y0) / a
        x_cruce = int(x0 - t * b)

        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)

        canvas = np.zeros_like(image)

        if x_cruce < mid and not left_drawn:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 5)
            canvas = cv2.bitwise_and(canvas, roi_3ch)
            cv2.add(result, canvas, result)
            left_drawn = True

        elif x_cruce >= mid and not right_drawn:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 5)
            canvas = cv2.bitwise_and(canvas, roi_3ch)
            cv2.add(result, canvas, result)
            right_drawn = True

    return result

# ================================================================
# 9. PIPELINE COMPLETO
# ================================================================
def process_frame(frame):
    filtered_image, mask_combined, mask_white, mask_yellow = hsv_color_filter(frame)

    gray    = to_grayscale(filtered_image)

    blurred = gaussian_blur(gray, kernel_size=(9, 9))

    edges   = canny_edge_detection(blurred)

    masked_edges, roi_vertices, roi_mask = region_of_interest(edges, frame.shape)

    lines = hough_lines(masked_edges, rho=1, theta=np.pi / 180, threshold=80)

    result = draw_lines_curvas(frame, lines, roi_mask, frame.shape)
    cv2.polylines(result, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=2)

    debug_images = {
        "filtered_bgr":   filtered_image,
        "mask_white":     mask_white,
        "mask_yellow":    mask_yellow,
        "mask_combined":  mask_combined,
        "grayscale":      gray,
        "gaussian":       blurred,
        "canny":          edges,
        "filtered_edges": masked_edges,
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
        "Mejora 1 — Filtro de Color HSV + ROI ampliada para curvas",
        fontsize=10, fontweight='bold'
    )

    frame_rgb    = cv2.cvtColor(frame,                        cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(debug_images["filtered_bgr"], cv2.COLOR_BGR2RGB)
    result_rgb   = cv2.cvtColor(debug_images.get("result_bgr", frame), cv2.COLOR_BGR2RGB)

    titles = [
        "1. Original (BGR)",
        "2. Máscara blanco (V≥120, S≤35)",
        "3. Máscara amarillo (H15-35, S≥35, V≥80)",
        "4. Imagen filtrada HSV + anti-veg",
        "5. Grises → Gaussiano 9×9",
        "6. Canny 30/90",
        "7. Bordes en ROI ampliada",
        "8. Resultado Hough (1 línea/lado)",
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
        ax.set_title(title, fontsize=9)
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
            cv2.imshow("Mejora 1 curvas", result)
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
    print("  MEJORA 1 ADAPTADA A CURVAS CON SOMBRA")
    print("  HSV recalibrado + anti-veg + Canny 30/90")
    print("  ROI amplia + HoughLines thr=80 + 1 línea/lado")
    print("=" * 60)
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encontró el video en '{VIDEO_PATH}'")
        sys.exit(1)
    process_video(video_path=VIDEO_PATH, show_preview=True, save_video=True)
