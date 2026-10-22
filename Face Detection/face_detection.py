import os
import threading
import time

# Matikan hardware transform Media Foundation (Windows) yang sering membuat webcam
# gagal streaming (error -1072875772) atau macet. Harus diset sebelum import cv2.
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')

import cv2
import mediapipe as mp
import numpy as np

# Step 1: Import kelas-kelas Face Detector dari MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'blaze_face_short_range.tflite')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f'Model tidak ditemukan: {MODEL_PATH}')

# Nama 6 keypoint wajah sesuai urutan output model
KEYPOINT_NAMES = ['Mata Kanan', 'Mata Kiri', 'Hidung', 'Mulut', 'Telinga Kanan', 'Telinga Kiri']

# Step 2: Callback untuk menerima hasil deteksi secara asynchronous
# Pada mode LIVE_STREAM, detect_async() langsung kembali tanpa menunggu,
# lalu hasil deteksi dikirim ke fungsi ini setiap kali satu frame selesai diproses
latest = {'result': None}
lock = threading.Lock()


def save_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    with lock:
        latest['result'] = result


def draw_detections(image, result):
    h, w = image.shape[:2]
    for detection in result.detections:
        # Bounding box dalam koordinat piksel
        bbox = detection.bounding_box
        start = (bbox.origin_x, bbox.origin_y)
        end = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start, end, (0, 0, 255), 2)

        # Skor kepercayaan deteksi
        score = detection.categories[0].score
        cv2.putText(image, f'Face {score:.2f}', (bbox.origin_x, max(bbox.origin_y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 6 keypoint dalam koordinat ternormalisasi [0, 1] -> konversi ke piksel
        for keypoint in detection.keypoints:
            cx, cy = int(keypoint.x * w), int(keypoint.y * h)
            cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)
    return image


# Step 3: Konfigurasi Face Detector dengan mode LIVE_STREAM
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_detection_confidence=0.5,
    min_suppression_threshold=0.3,
    result_callback=save_result)

WINDOW_NAME = 'Face Detection'
VIDEO_FPS = 20.0


def open_camera(retries=10):
    # Buka kamera default (index 0); coba ulang jika kamera masih sibuk / belum siap
    # Di Windows pakai backend MSMF secara langsung (lebih stabil dan tanpa pesan error obsensor)
    api = cv2.CAP_MSMF if os.name == 'nt' else cv2.CAP_ANY
    for _ in range(retries):
        cam = cv2.VideoCapture(0, api)
        # Kamera dianggap siap jika sudah bisa membaca satu frame
        if cam.isOpened() and cam.read()[0]:
            return cam
        cam.release()
        time.sleep(2)
    raise RuntimeError('Kamera tidak dapat dibuka')


# Step 4: Buka kamera dan siapkan VideoWriter untuk merekam hasil deteksi
cam = open_camera()
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(OUTPUT_DIR, 'face_detection.mp4'), fourcc, VIDEO_FPS,
                      (frame_width, frame_height))

with FaceDetector.create_from_options(options) as detector:
    start_time = time.monotonic()
    last_ts = -1
    prev_time = start_time
    next_write = None
    fps = 0.0
    fails = 0
    prev_frame = None
    annotated = None

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                # Kamera kadang terputus sesaat: buka ulang, menyerah setelah 5 kali gagal
                fails += 1
                if fails > 5:
                    print('Frame tidak dapat dibaca, keluar dari loop')
                    break
                print('Frame gagal dibaca, membuka ulang kamera...')
                cam.release()
                time.sleep(2)
                cam = open_camera()
                continue
            fails = 0
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Pada cahaya redup webcam hanya mengirim beberapa frame baru per detik, dan
            # read() mengembalikan frame yang sama berulang kali. Frame duplikat tidak
            # perlu dideteksi ulang dan tidak dihitung dalam FPS
            is_new = prev_frame is None or not np.array_equal(frame, prev_frame)
            prev_frame = frame
            now = time.monotonic()

            if is_new:
                # Step 5: Konversi frame BGR (OpenCV) ke RGB lalu ke mediapipe.Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Step 6: Kirim frame ke detector beserta timestamp (ms, harus selalu naik)
                timestamp_ms = max(int((now - start_time) * 1000), last_ts + 1)
                last_ts = timestamp_ms
                detector.detect_async(mp_image, timestamp_ms)

                # FPS kamera (frame baru per detik), dihaluskan dengan moving average
                current_fps = 1.0 / max(now - prev_time, 1e-3)
                fps = current_fps if fps == 0.0 else 0.9 * fps + 0.1 * current_fps
                prev_time = now

            # Step 7: Gambar hasil deteksi terbaru pada frame
            with lock:
                result = latest['result']
            annotated = frame.copy()
            num_faces = 0
            if result is not None:
                draw_detections(annotated, result)
                num_faces = len(result.detections)

            cv2.putText(annotated, f'FPS: {fps:.1f}  Wajah: {num_faces}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Tulis ke video dengan laju tetap VIDEO_FPS; jumlah frame mengikuti waktu nyata
            # (frame diulang jika kamera lebih lambat) agar durasinya sesuai rekaman
            if next_write is None:
                next_write = now
            while next_write <= now:
                out.write(annotated)
                next_write += 1.0 / VIDEO_FPS
            cv2.imshow(WINDOW_NAME, annotated)

            # Tekan 's' untuk menyimpan screenshot, 'q' (atau tutup jendela) untuk keluar
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                path = os.path.join(OUTPUT_DIR, f'screenshot_{int(time.time())}.png')
                cv2.imwrite(path, annotated)
                print(f'Screenshot disimpan: {path}')
            elif key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print('Dihentikan dengan Ctrl+C')

    # Step 8: Simpan frame terakhir dan cetak hasil deteksi terakhir
    if annotated is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'snapshot.png'), annotated)
    with lock:
        result = latest['result']
    if result is not None:
        for i, detection in enumerate(result.detections):
            bbox = detection.bounding_box
            print(f'Detection #{i}: bbox=({bbox.origin_x}, {bbox.origin_y}, '
                  f'{bbox.width}, {bbox.height}) score={detection.categories[0].score:.4f}')
            for name, kp in zip(KEYPOINT_NAMES, detection.keypoints):
                print(f'    {name:<14} x={kp.x:.4f} y={kp.y:.4f}')

cam.release()
out.release()
cv2.destroyAllWindows()
print(f'Hasil disimpan di: {OUTPUT_DIR}')
