# Datasheet Dataset Balok

## Gambaran Umum
| Item | Keterangan |
|---|---|
| Tugas | Klasifikasi citra (5 kelas) |
| Jumlah gambar | 600 (120 per kelas, seimbang) |
| Format | JPG, 640×480 px |
| Sumber | Foto kamera robot/praktikum (isi sumber, alat, dan tanggal pengambilan) |
| Lisensi | (isi, mis. CC BY 4.0 atau "hanya untuk keperluan akademik") |

## Kelas
| Kelas | Jumlah | Deskripsi |
|---|---:|---|
| `all_blocks` | 120 | Beberapa balok berbagai warna dalam satu gambar |
| `balok_hijautua` | 120 | Satu balok hijau tua |
| `balok_hitam` | 120 | Satu balok hitam |
| `balok_merahmuda` | 120 | Satu balok merah/merah muda |
| `papan_balok` | 120 | Papan hitam bertitik |

## Pembagian Data
| Split | Per kelas | Total |
|---|---:|---:|
| train | 84 | 420 |
| val | 18 | 90 |
| test | 18 | 90 |

Struktur folder mengikuti format `torchvision.datasets.ImageFolder`:

```
dataset/
├── train/<kelas>/*.jpg
├── val/<kelas>/*.jpg
└── test/<kelas>/*.jpg
```
Data mentah sebelum dibagi ada di `dataset_raw/` (tidak diikutkan di repo).

## Penamaan File
`<kelas>_<nomor 3 digit>.jpg`, contoh `balok_hitam_014.jpg`.

## Catatan Penggunaan
- Gambar diubah ke 224×224 dan dinormalisasi dengan mean/std ImageNet saat training.
- Augmentasi hanya dikenakan pada data train.
- Latar belakang dan pencahayaan bervariasi antar foto (lantai abu-abu, bayangan, dan perbedaan warna cahaya).
- Dataset kecil dan satu domain; belum teruji untuk objek, kamera, atau lingkungan lain.
- Belum diketahui apakah foto yang mirip (frame berurutan) tersebar lintas split. Jika ya, akurasi test bisa terlalu optimistis.
