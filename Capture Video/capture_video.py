import os
import time

# Matikan hardware transform Media Foundation (Windows) yang sering membuat webcam
# gagal streaming (error -1072875772) atau macet. Harus diset sebelum import cv2.
os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS', '0')

import cv2

WINDOW_NAME = 'Camera'
VIDEO_FPS = 20.0

# Step 1: Persiapan folder output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def open_camera(retries=10):
    # Buka kamera default (index 0); coba ulang jika kamera masih sibuk / belum siap.
    # Jika perangkat memiliki lebih dari satu kamera, gunakan index 1, 2, dst.
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


# Step 2: Buka kamera
cam = open_camera()

# Step 3: Ambil lebar dan tinggi frame default dari kamera
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Resolusi kamera: {frame_width}x{frame_height}')

# Step 4: Definisikan codec dan buat objek VideoWriter
# 'mp4v' = codec MPEG-4, 20.0 = frame rate, (lebar, tinggi) = dimensi video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = os.path.join(OUTPUT_DIR, 'output.mp4')
out = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (frame_width, frame_height))

# Step 5: Baca frame dari kamera secara terus-menerus
# read() mengembalikan ret (bool, berhasil/tidak) dan frame (array numpy 3D)
frame = None
fails = 0
next_write = None
try:
    while True:
        ret, current = cam.read()
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
        frame = cv2.resize(current, (frame_width, frame_height))

        # Tulis frame ke file video output dengan laju tetap VIDEO_FPS.
        # Jumlah frame yang ditulis mengikuti waktu nyata (frame diulang jika kamera
        # lebih lambat) agar durasi video sama dengan durasi rekaman sebenarnya
        now = time.monotonic()
        if next_write is None:
            next_write = now
        while next_write <= now:
            out.write(frame)
            next_write += 1.0 / VIDEO_FPS

        # Tampilkan frame yang ditangkap
        cv2.imshow(WINDOW_NAME, frame)

        # Tekan 'q' (atau tutup jendela) untuk keluar dari loop
        key = cv2.waitKey(1)
        window_closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        if key == ord('q') or window_closed:
            break
except KeyboardInterrupt:
    print('Dihentikan dengan Ctrl+C')

# Step 6: Simpan frame terakhir sebagai snapshot
if frame is not None:
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'snapshot.png'), frame)

# Step 7: Lepaskan objek capture dan writer
cam.release()
out.release()
cv2.destroyAllWindows()
print(f'Hasil disimpan di: {OUTPUT_DIR}')
