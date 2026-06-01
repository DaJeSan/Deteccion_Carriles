import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import deque

# ================================================================
# RUTAS
# ================================================================
VIDEO_PATH        = os.path.join("imagenes", "video_carretera.mp4")
OUTPUT_VIDEO_PATH = "resultado_mejora3_kalman.avi"
DIAGRAMA_PATH     = "pipeline_mejora3_steps.png"

# ================================================================
# PARÁMETROS GLOBALES DE ESTABILIZACIÓN TEMPORAL  ← MEJORA 3
# ================================================================
HISTORY_N = 8
OUTLIER_SIGMA = 2.0

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
# 4. DETECTOR DE BORDES CANNY  (sin cambios)
# ================================================================
def canny_edge_detection(blurred, low_threshold=50, high_threshold=150):
    return cv2.Canny(blurred, low_threshold, high_threshold)

# ================================================================
# 5. FILTRO DE COLOR HSV  (sin cambios)
# ================================================================
def hsv_color_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([ 18,  40, 100], dtype=np.uint8)
    upper_yellow = np.array([ 30, 255, 255], dtype=np.uint8)
    mask_yellow  = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white  = np.array([  0,   0, 145], dtype=np.uint8)
    upper_white  = np.array([179,  40, 255], dtype=np.uint8)
    mask_white   = cv2.inRange(hsv, lower_white, upper_white)

    mask_combined  = cv2.bitwise_or(mask_yellow, mask_white)
    filtered_image = cv2.bitwise_and(image, image, mask=mask_combined)

    return filtered_image, mask_combined, mask_white, mask_yellow

# ================================================================
# 6. REGIÓN DE INTERÉS  (sin cambios)
# ================================================================
def region_of_interest(image, image_shape):
    height, width = image_shape[0], image_shape[1]
    bottom_left  = (int(0.15 * width),  int(0.85 * height))
    bottom_right = (int(0.85 * width),  int(0.85 * height))
    top_left     = (int(0.425 * width), int(0.55 * height))
    top_right    = (int(0.575 * width), int(0.55 * height))
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image, vertices, mask

# ================================================================
# 7. HOUGH PROBABILÍSTICA CALIBRADA  (sin cambios, Mejora 2)
# ================================================================
def hough_lines_probabilistic(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        min_line_length=25,
        max_line_gap=80):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    return lines

# ================================================================
# 8. CLASIFICAR SEGMENTOS EN CARRIL IZQUIERDO / DERECHO
# ================================================================
def classify_segments(lines, image_width):
    left_lines  = []
    right_lines = []

    if lines is None:
        return left_lines, right_lines

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope     = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if abs(slope) < 0.3 or abs(slope) > 5.0:
            continue

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    return left_lines, right_lines

# ================================================================
# 9. FILTRO DE KALMAN POR CARRIL  ← MEJORA 3
# ================================================================
class LaneKalmanFilter:

    def __init__(self):
        # ── Estado inicial ──────────────────────────────────────
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1000.0
        # ── Modelo de transición: identidad ─────────────────────
        self.F = np.eye(2, dtype=np.float64)
        # ── Matriz de observación: identidad ────────────────────
        self.H = np.eye(2, dtype=np.float64)
        # ── Ruido de proceso Q ──────────────────────────────────
        self.Q = np.diag([1e-4, 10.0])
        # ── Ruido de medición R ─────────────────────────────────
        self.R = np.diag([1e-2, 100.0])
        # ── Estado de inicialización ────────────────────────────
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, measurement):
        z = measurement.reshape(2, 1)

        if not self.initialized:
            self.x = z.copy()
            self.initialized = True
            return self.x.flatten()

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

        return self.x.flatten()

# ================================================================
# 10. BUFFER DE HISTORIAL + DETECCIÓN DE OUTLIERS  ← MEJORA 3
# ================================================================
class LaneHistory:

    def __init__(self, maxlen=HISTORY_N):
        self.history = deque(maxlen=maxlen)

    def is_outlier(self, measurement, sigma=OUTLIER_SIGMA):
        if len(self.history) < 3:
            return False

        hist = np.array(self.history)
        mean = hist.mean(axis=0)
        std  = hist.std(axis=0) + 1e-6

        distance = np.abs(measurement - mean) / std
        return bool(np.any(distance > sigma))

    def add(self, measurement):
        self.history.append(measurement.copy())

    def last_valid(self):
        if self.history:
            return self.history[-1]
        return None

# ================================================================
# 11. ESTABILIZADOR TEMPORAL COMPLETO (Kalman + Buffer)  ← MEJORA 3
# ================================================================
class LaneStabilizer:

    def __init__(self, maxlen=HISTORY_N):
        self.kf      = LaneKalmanFilter()
        self.history = LaneHistory(maxlen=maxlen)

    def _weighted_average(self, line_params, lines_raw):
        weights = []
        for x1, y1, x2, y2 in lines_raw:
            length = np.hypot(x2 - x1, y2 - y1)
            weights.append(max(length, 1.0))

        total_w = sum(weights)
        slopes     = [p[0] * w for p, w in zip(line_params, weights)]
        intercepts = [p[1] * w for p, w in zip(line_params, weights)]

        return np.array([sum(slopes) / total_w, sum(intercepts) / total_w])

    def update(self, line_params, lines_raw):
        self.kf.predict()

        if not line_params:
            if self.kf.initialized:
                return self.kf.x.flatten()
            return None

        measurement = self._weighted_average(line_params, lines_raw)

        if self.history.is_outlier(measurement):
            if self.kf.initialized:
                return self.kf.x.flatten()
            return None

        self.history.add(measurement)
        estimated = self.kf.update(measurement)
        return estimated

# ================================================================
# 12. CONVERTIR (slope, intercept) A COORDENADAS DE IMAGEN
# ================================================================
def lane_to_coords(slope, intercept, image_shape):
    height = image_shape[0]
    y_bottom = int(0.85 * height)
    y_top    = int(0.55 * height)

    if abs(slope) < 1e-6:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top    = int((y_top    - intercept) / slope)

    return (x_bottom, y_bottom), (x_top, y_top)

# ================================================================
# 13. DIBUJAR LOS CARRILES ESTABILIZADOS
# ================================================================
def draw_stabilized_lanes(image, left_state, right_state, image_shape,
                          color=(0, 255, 0), thickness=6):
    result = image.copy()

    for state in [left_state, right_state]:
        if state is None:
            continue
        slope, intercept = state
        coords = lane_to_coords(slope, intercept, image_shape)
        if coords is None:
            continue
        pt_bottom, pt_top = coords
        h, w = image_shape[:2]
        if not (0 <= pt_bottom[0] < w and 0 <= pt_top[0] < w):
            continue
        cv2.line(result, pt_bottom, pt_top, color, thickness)

    return result

# ================================================================
# 14. PIPELINE COMPLETO — MEJORA 3
# ================================================================
def process_frame(frame, left_stabilizer, right_stabilizer):
    roi_frame, roi_vertices, roi_mask = region_of_interest(frame, frame.shape)

    filtered_image, mask_combined, mask_white, mask_yellow = hsv_color_filter(roi_frame)

    gray    = to_grayscale(filtered_image)

    blurred = gaussian_blur(gray, kernel_size=(9, 9))

    edges   = canny_edge_detection(blurred, low_threshold=50, high_threshold=150)

    lines = hough_lines_probabilistic(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        min_line_length=40,
        max_line_gap=80
    )

    left_params, right_params = classify_segments(lines, frame.shape[1])

    left_raw  = []
    right_raw = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3 or abs(slope) > 5.0:
                continue
            if slope < 0:
                left_raw.append((x1, y1, x2, y2))
            else:
                right_raw.append((x1, y1, x2, y2))

    left_state  = left_stabilizer.update(left_params,  left_raw)
    right_state = right_stabilizer.update(right_params, right_raw)

    result = draw_stabilized_lanes(
        frame, left_state, right_state, frame.shape,
        color=(0, 255, 0), thickness=6
    )
    cv2.polylines(result, roi_vertices, isClosed=True, color=(0, 0, 255), thickness=2)

    debug_images = {
        "roi_frame":      roi_frame,
        "filtered_bgr":   filtered_image,
        "mask_white":     mask_white,
        "mask_yellow":    mask_yellow,
        "mask_combined":  mask_combined,
        "grayscale":      gray,
        "gaussian":       blurred,
        "canny":          edges,
        "filtered_edges": edges,
        "roi_mask":       roi_mask,
        "roi_vertices":   roi_vertices,
    }
    return result, debug_images, left_state, right_state

# ================================================================
# 15. DIAGRAMA DEL PIPELINE
# ================================================================
def show_pipeline_diagram(frame, debug_images):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig.suptitle(
        "Pipeline Mejora 3 — Estabilización temporal (Kalman + buffer deque)",
        fontsize=11, fontweight='bold'
    )

    frame_rgb    = cv2.cvtColor(frame,                        cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(debug_images["filtered_bgr"], cv2.COLOR_BGR2RGB)
    result_rgb   = cv2.cvtColor(debug_images.get("result_bgr", frame), cv2.COLOR_BGR2RGB)

    titles = [
        "1. Original (BGR)",
        "2. ROI sobre original",
        "3. Máscara blanco",
        "4. Máscara amarillo",
        "5. Imagen filtrada HSV",
        "6. Grises → Gaussiano",
        "7. Canny 50/150",
        "8. Resultado Kalman estabilizado",
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
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(DIAGRAMA_PATH, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Diagrama guardado en '{DIAGRAMA_PATH}'")

# ================================================================
# 16. PROCESAR VIDEO COMPLETO
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

    left_stabilizer  = LaneStabilizer(maxlen=HISTORY_N)
    right_stabilizer = LaneStabilizer(maxlen=HISTORY_N)

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

        result, debug_images, left_state, right_state = process_frame(
            frame, left_stabilizer, right_stabilizer
        )
        debug_images["result_bgr"] = result

        if not first_done:
            show_pipeline_diagram(frame, debug_images)
            first_done = True

        if writer is not None:
            writer.write(result)

        if show_preview:
            if left_state is not None:
                txt = f"Izq  slope={left_state[0]:.3f}  ic={left_state[1]:.1f}"
                cv2.putText(result, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            if right_state is not None:
                txt = f"Der  slope={right_state[0]:.3f}  ic={right_state[1]:.1f}"
                cv2.putText(result, txt, (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            cv2.imshow("Mejora 3 — Kalman estabilizado", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"Interrumpido en frame {frame_count}")
                break

        if frame_count % 50 == 0 or frame_count == 1:
            sl = f"{left_state[0]:.3f}"  if left_state  is not None else "—"
            sr = f"{right_state[0]:.3f}" if right_state is not None else "—"
            print(f"  Frame {frame_count}/{total}  "
                  f"slope_izq={sl}  slope_der={sr}")

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
    print("  MEJORA 3: Estabilización temporal Kalman + buffer deque")
    print("  Estado: [pendiente, intercepto]  F=I  Q=diag[1e-4,10]")
    print("  Buffer N=8 frames   Outlier sigma=2.0")
    print("=" * 60)
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: No se encontró el video en '{VIDEO_PATH}'")
        sys.exit(1)
    process_video(video_path=VIDEO_PATH, show_preview=True, save_video=True)
