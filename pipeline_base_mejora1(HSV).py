import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# RUTAS

VIDEO_PATH = os.path.join("imagenes", "video_carretera.mp4")
OUTPUT_VIDEO_PATH = "resultado_mejora1_hsv.avi"
DIAGRAMA_PATH = "pipeline_mejora1_steps.png"

# CARGA DE VIDEO

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")
    return cap

# ESCALA DE GRISES

def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# FILTRO GAUSSIANO

def gaussian_blur(gray, kernel_size=(9, 9)):
    return cv2.GaussianBlur(gray, kernel_size, sigmaX=0)


#DETECTOR DE BORDES CANNY

def canny_edge_detection(blurred, low_threshold=50, high_threshold=150):
    
    return cv2.Canny(blurred, low_threshold, high_threshold)

# FILTRO DE COLOR HSV (BLANCO + AMARILLO)

def hsv_color_filter(image):
  
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Amarillo 
    lower_yellow = np.array([ 18,  40, 100], dtype=np.uint8)
    upper_yellow = np.array([ 30, 255, 255], dtype=np.uint8)  
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Blanco
    lower_white  = np.array([  0,   0, 170], dtype=np.uint8)  
    upper_white  = np.array([179,  40, 255], dtype=np.uint8)  
    mask_white   = cv2.inRange(hsv, lower_white, upper_white)

    #  Unión de mascaras
    mask_combined   = cv2.bitwise_or(mask_yellow, mask_white)
    filtered_image  = cv2.bitwise_and(image, image, mask=mask_combined)

    return filtered_image, mask_combined, mask_white, mask_yellow

# REGIÓN DE INTERÉS (ROI) (MÁSCARA TRAPEZOIDAL)

def region_of_interest(image, image_shape):
    height, width = image_shape[0], image_shape[1]
    bottom_left  = (int(0.15 * width), int(0.85 * height))
    bottom_right = (int(0.85 * width), int(0.85 * height))
    top_left     = (int(0.425 * width), int(0.55 * height))
    top_right    = (int(0.575 * width), int(0.55 * height))
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]],
                        dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, vertices, mask


# TRANSFORMADA DE HOUGH 

def hough_lines(masked_edges, rho=1, theta=np.pi / 180, threshold=60):

    lines = cv2.HoughLines(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold
    )
    return lines

# DIBUJAR LÍNEAS SOBRE LA IMAGEN ORIGINAL

def draw_lines(image, lines, roi_mask, color=(0, 255, 0), thickness=4):
 
    result = image.copy()
    if lines is None or len(lines) == 0:
        return result

    line_canvas = np.zeros_like(image)
    for line in lines:
        rho_val, theta_val = line[0]
        angle_deg = abs(np.degrees(theta_val))

        # Descarte de líneas verticales extremas
        if angle_deg < 20 or angle_deg > 160:
            continue
        # Descarte de líneas casi horizontales 
        if abs(angle_deg - 90) < 10:
            continue

        a  = np.cos(theta_val)
        b  = np.sin(theta_val)
        x0 = a * rho_val
        y0 = b * rho_val
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * a)
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * a)
        cv2.line(line_canvas, (x1, y1), (x2, y2), color, thickness)

    roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    line_canvas  = cv2.bitwise_and(line_canvas, roi_mask_3ch)
    cv2.add(result, line_canvas, result)
    return result

# PIPELINE COMPLETO PARA UN SOLO FRAME

def process_frame(frame):

    # Paso 1: ROI sobre original
    roi_frame, roi_vertices, roi_mask = region_of_interest(frame, frame.shape)

    # Paso 2: Filtro HSV sobre ROI
    filtered_image, mask_combined, mask_white, mask_yellow = hsv_color_filter(roi_frame)

    # Paso 3: Grises de la imagen ya filtrada
    gray    = to_grayscale(filtered_image)

    # Paso 4: Suavizado
    blurred = gaussian_blur(gray, kernel_size=(9, 9))

    # Paso 5: Bordes — umbrales 50/150
    edges   = canny_edge_detection(blurred, low_threshold=50, high_threshold=150)

    # Paso 6: AND bordes ∩ máscara color
    filtered_edges = cv2.bitwise_and(edges, mask_combined)

    # Paso 7: Hough estándar — threshold=60
    lines = hough_lines(filtered_edges, rho=1, theta=np.pi / 180, threshold=60)

    # Paso 8: Dibujo
    result = draw_lines(frame, lines, roi_mask, color=(0, 255, 0), thickness=4)
    cv2.polylines(result, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=2)

    debug_images = {
        "roi_frame":     roi_frame,
        "filtered_bgr":  filtered_image,
        "mask_white":    mask_white,
        "mask_yellow":   mask_yellow,
        "mask_combined": mask_combined,
        "grayscale":     gray,
        "gaussian":      blurred,
        "canny":         edges,
        "filtered_edges":filtered_edges,
        "roi_mask":      roi_mask,
        "roi_vertices":  roi_vertices,
    }
    return result, debug_images

# 10. DIAGRAMA DEL PIPELINE 

def show_pipeline_diagram(frame, debug_images):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle("Pipeline Mejora 1 (Filtro de Color HSV)",
                 fontsize=16, fontweight='bold')

    frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(debug_images["filtered_bgr"], cv2.COLOR_BGR2RGB)
    result_rgb   = cv2.cvtColor(debug_images.get("result_bgr", frame), cv2.COLOR_BGR2RGB)

    titles = [
        "1. Original (BGR)",
        "2. ROI sobre original",
        "3. Máscara blanco (HSV)",
        "4. Máscara amarillo (HSV)",
        "5. Imagen filtrada (B+A)",
        "6. Grises → Gaussiano",
        "7. Canny 50/150",
        "8. Resultado (Hough)",
    ]
    images = [
        frame_rgb,
        cv2.cvtColor(debug_images["roi_frame"], cv2.COLOR_BGR2RGB),
        debug_images["mask_white"],
        debug_images["mask_yellow"],
        filtered_rgb,
        debug_images["gaussian"],
        debug_images["canny"],
        result_rgb,
    ]
    cmaps = [None, None, 'gray', 'gray', None, 'gray', 'gray', None]

    for ax, img, title, cmap in zip(axes.flat, images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(DIAGRAMA_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Diagrama guardado en '{DIAGRAMA_PATH}'")

# PROCESAR VIDEO COMPLETO
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
            cv2.imshow("Mejora 1 HSV", result)
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

# EJECUCIÓN PRINCIPAL

if __name__ == "__main__":
    print("=" * 60)
    print("  MEJORA 1: Filtro de Color HSV")
    print("  Flujo: HSV → Gris → Gauss → Canny → ROI → Hough")
    print("=" * 60)
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encontró el video en '{VIDEO_PATH}'")
        sys.exit(1)
    process_video(video_path=VIDEO_PATH, show_preview=True, save_video=True)