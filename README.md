# Praktikum Pengolahan Citra Digital - Video Capture & Face Detection

Repository ini berisi implementasi **pengambilan video dari kamera (webcam)** menggunakan OpenCV dan **deteksi wajah secara real-time** menggunakan **MediaPipe Face Detector** dalam mode *live stream*.

Sumber referensi:
* [Python OpenCV: Capture Video from Camera — GeeksforGeeks](https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/)
* [Face detection guide for Python — Google AI Edge MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python#live-stream)

---

## 📌 Daftar Program

### 1. Capture Video (`cv2.VideoCapture` + `cv2.VideoWriter`)
Langkah-langkah:
1. `cv2.VideoCapture(0)` membuat objek kamera. Argumen `0` adalah index kamera (kamera pertama); gunakan `1` untuk kamera kedua.
2. `cam.get(cv2.CAP_PROP_FRAME_WIDTH)` dan `cam.get(cv2.CAP_PROP_FRAME_HEIGHT)` mengambil dimensi frame default kamera.
3. `cv2.VideoWriter` menyimpan frame menjadi file video dengan parameter:
   * **Filename** — `output/output.mp4`
   * **FourCC** — kode 4 karakter codec, di sini `'mp4v'` (MPEG-4). Contoh lain: `'MJPG'` (Motion JPEG), `'X264'` (H.264).
   * **Frame rate** — `20.0` fps. FPS lebih tinggi menghasilkan video lebih halus tetapi ukuran file lebih besar.
   * **Dimensi video** — tuple `(lebar, tinggi)`.
4. Di dalam *infinite loop*, `cam.read()` mengembalikan `ret` (boolean berhasil/tidak) dan `frame` (array numpy 3D: tinggi × lebar × 3 kanal warna BGR).
5. `out.write(frame)` menulis frame ke file video, `cv2.imshow()` menampilkannya.
6. `cv2.waitKey(1)` menunggu input keyboard selama 1 ms; jika tombol `q` ditekan (`ord('q')` = nilai ASCII), loop berhenti.
7. `cam.release()`, `out.release()`, dan `cv2.destroyAllWindows()` melepaskan seluruh resource.

### 2. Face Detection (MediaPipe Tasks — `FaceDetector`, mode `LIVE_STREAM`)
Langkah-langkah:
1. **Model** — menggunakan model `blaze_face_short_range.tflite` (BlazeFace, dioptimalkan untuk wajah dalam jarak ±2 meter dari kamera, cocok untuk webcam).
2. **Create the task** — `FaceDetector.create_from_options(options)` dengan konfigurasi:

   | Opsi | Nilai | Keterangan |
   | :--- | :---: | :--- |
   | `running_mode` | `LIVE_STREAM` | Mode untuk input kamera; hasil diterima secara *asynchronous* |
   | `min_detection_confidence` | `0.5` | Skor minimum agar deteksi dianggap berhasil |
   | `min_suppression_threshold` | `0.3` | Ambang *non-maximum suppression* untuk deteksi yang tumpang tindih |
   | `result_callback` | `save_result` | Fungsi *listener* yang menerima hasil deteksi |

3. **Prepare data** — frame OpenCV (BGR) dikonversi ke RGB, lalu dibungkus menjadi `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`.
4. **Run the task** — `detector.detect_async(mp_image, timestamp_ms)`. Pada mode live stream fungsi ini **tidak memblokir** thread; hasil dikirim ke callback setiap kali satu frame selesai diproses. Timestamp wajib selalu bertambah. Jika detector masih sibuk, frame baru akan diabaikan.
5. **Handle and display results** — `FaceDetectorResult` berisi daftar `detections`, masing-masing memiliki:
   * `bounding_box` (`origin_x`, `origin_y`, `width`, `height`) dalam piksel → digambar sebagai kotak merah.
   * `categories[0].score` → skor kepercayaan, ditampilkan di atas kotak.
   * 6 `keypoints` ternormalisasi `[0, 1]`: mata kanan, mata kiri, ujung hidung, mulut, tragion telinga kanan, tragion telinga kiri → digambar sebagai titik hijau.

Pada mode video/live stream, Face Detector memakai *tracking* sehingga model deteksi tidak dijalankan pada setiap frame, yang membantu menurunkan latensi.

---

## 🖼️ Hasil Visualisasi

### 1. Capture Video
![Capture Video](Capture%20Video/output/snapshot.png)

### 2. Face Detection
![Face Detection](Face%20Detection/output/snapshot.png)

> Gambar di atas dihasilkan otomatis (frame terakhir) setelah script dijalankan dan ditutup dengan tombol `q`.

---

## 📂 Struktur Direktori

```text
├── Capture Video/
│   ├── capture_video.py               # Script capture video dari webcam
│   └── output/                        # output.mp4 & snapshot.png
├── Face Detection/
│   ├── face_detection.py              # Script deteksi wajah real-time (MediaPipe)
│   └── output/                        # face_detection.mp4, snapshot.png, screenshot_*.png
├── models/
│   └── blaze_face_short_range.tflite  # Model MediaPipe Face Detector
├── requirements.txt                   # Dependensi pustaka Python
├── .gitignore                         # Aturan ignore file git
└── README.md                          # Dokumentasi proyek
```

---

## 🚀 Cara Menjalankan

### 1. Prasyarat & Instalasi
Pastikan Python telah terpasang dan webcam tersedia. Install dependensi dengan perintah:
```bash
pip install -r requirements.txt
```

Model sudah disertakan di folder `models/`. Jika hilang, unduh ulang dengan:
```bash
curl -L -o models/blaze_face_short_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
```

### 2. Eksekusi Script
```bash
python "Capture Video/capture_video.py"
python "Face Detection/face_detection.py"
```

Kontrol keyboard (pada jendela video):
* `q` — keluar dan simpan hasil (menutup jendela dengan tombol X atau menekan `Ctrl+C` di terminal juga tetap menyimpan hasil)
* `s` — simpan screenshot (khusus Face Detection)

Video direkam dengan laju tetap 20 fps sehingga durasinya sesuai dengan lama rekaman. Jika kamera terputus sesaat, script otomatis membuka ulang kamera (maksimal 5 kali berturut-turut).

### 3. Catatan Kamera di Windows
* Script mematikan *hardware transform* Media Foundation (`OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`) dan memakai backend `cv2.CAP_MSMF` secara langsung. Tanpa ini, sebagian webcam gagal streaming dengan error `-1072875772` (`MF_E_HW_MFT_FAILED_START_STREAMING`) atau macet setelah dibuka.
* Setelah sebuah script ditutup, kamera kadang butuh beberapa detik sebelum bisa dibuka lagi; script akan menunggu dan mencoba ulang secara otomatis.
* Pada cahaya redup webcam menurunkan laju frame (±7 fps) untuk memperpanjang eksposur. `read()` lalu mengembalikan frame yang sama berulang kali; frame duplikat ini dilewati sehingga FPS yang ditampilkan adalah FPS kamera yang sebenarnya. Gunakan ruangan yang lebih terang untuk FPS lebih tinggi.

Video hasil rekaman dan snapshot frame terakhir tersimpan di folder `output/` masing-masing. Script Face Detection juga mencetak *bounding box*, skor, dan koordinat 6 keypoint dari deteksi terakhir ke terminal.

---

## 👤 Penulis
* **Amirullah**
* Praktikum Pengolahan Citra Digital
* Politeknik Negeri Lhokseumawe (PNL)
