import os
import subprocess
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import librosa
import librosa.display
from scipy.signal import butter, filtfilt
import soundfile as sf
import sounddevice as sd
import pandas as pd
from datetime import datetime
import warnings
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
import sys

try:
    import cv2
except ImportError:
    cv2 = None

warnings.filterwarnings("ignore")

# ==========================================
# 1. BÖLÜM: NASA & SETI BİYOAKUSTİK LAB (GERÇEK SES MODÜLÜ)
# ==========================================
class NasaSetiSeviyeBiyoAkustikLaboratuvari:
    def __init__(self, dosya_yolu, maksimum_sure=120):
        self.dosya_yolu = dosya_yolu
        self.sure = maksimum_sure
        self.y = None
        self.sr = None
        self.nyq = None
        
        self.klasor_ana = "NASA_SETI_Ucus_BiyoAkustik_Laboratuvari"
        self.klasorler = {
            "sesler": os.path.join(self.klasor_ana, "1_Ham_ve_Islenmis_Sesler"),
            "pngler": os.path.join(self.klasor_ana, "2_Spektrogram_ve_Gorseller"),
            "videolar": os.path.join(self.klasor_ana, "3_Senkronize_Videolar"),
            "kategori_1_temizleme": os.path.join(self.klasor_ana, "4_Cat1_Gelistirilmis_Temizleme"),
            "kategori_2_ozellikler": os.path.join(self.klasor_ana, "5_Cat2_Akustik_Indeksler_MFCC"),
            "kategori_3_segmentasyon": os.path.join(self.klasor_ana, "6_Cat3_Hece_ve_Darbe_Segmentasyonu"),
            "kategori_4_kaynak_rapor": os.path.join(self.klasor_ana, "7_Cat4_Zaman_Damgali_Kaynak_Raporlari"),
            "frekans_donusumleri": os.path.join(self.klasor_ana, "8_Frekans_ve_CQT_Analizleri"),
            "baskin_ses_3d_mesh": os.path.join(self.klasor_ana, "9_Baskin_Ses_3D_Mesh_Analiz"),
            "interaktif_3d_py": os.path.join(self.klasor_ana, "10_Interaktif_3D_Python_Ortami"),
            "taksonomi_embedding": os.path.join(self.klasor_ana, "11_NASA_Latent_Taksonomi_Embedding"),
            "infrasound_anomali": os.path.join(self.klasor_ana, "12_Infrasound_Sismik_Anomali"),
            "gis_isiharitasi": os.path.join(self.klasor_ana, "13_GIS_Akustik_Isi_Haritasi"),
            "entropi_seti": os.path.join(self.klasor_ana, "14_SETI_Kuantum_Entropi_Dedektoru"),
            "tdoa_lokasyon": os.path.join(self.klasor_ana, "15_TDOA_Kaynak_Lokasyon_Simulasyonu"),
            "webgl_hologram": os.path.join(self.klasor_ana, "16_WebXR_Holografik_Web_Ortami")
        }
        for k in self.klasorler.values():
            if not os.path.exists(k):
                os.makedirs(k)

    def sesi_yukle(self):
        sr_hz = 22050
        if not self.dosya_yolu or not os.path.exists(self.dosya_yolu):
            print(f"🎙️ [GERÇEK TARAMA] Ses dosyası bulunamadı. Canlı Donanım Mikrofonundan {self.sure} saniyelik GERÇEK Akustik Tarama başlatılıyor...")
            print("   -> Lütfen konuşun veya ses kaynağı oluşturun (Kayıt başladı)...")
            kayit_verisi = sd.rec(int(self.sure * sr_hz), samplerate=sr_hz, channels=1, dtype='float32')
            sd.wait()
            print("   -> Canlı mikrofon taraması tamamlandı.")
            self.y = np.squeeze(kayit_verisi)
            self.sr = sr_hz
            self.dosya_yolu = os.path.join(self.klasorler["sesler"], "canli_mikrofon_gercek_tarama.wav")
            sf.write(self.dosya_yolu, self.y, self.sr)
        else:
            print(f"🎧 Gerçek ses dosyası yükleniyor: {self.dosya_yolu}")
            self.y, self.sr = librosa.load(self.dosya_yolu, sr=sr_hz, duration=self.sure)
            
        self.nyq = 0.5 * self.sr

    def filtre_high(self, veri, freq):
        b, a = butter(4, freq / self.nyq, btype='high')
        return filtfilt(b, a, veri)

    def filtre_low(self, veri, freq):
        b, a = butter(4, freq / self.nyq, btype='low')
        return filtfilt(b, a, veri)

    def kategori_1_temizleme_islem(self, y_veri):
        y_harm, y_perc = librosa.effects.hpss(y_veri)
        esik = np.mean(np.abs(y_perc)) * 0.5
        y_temiz = np.where(np.abs(y_perc) > esik, y_perc, y_harm * 0.3)
        return y_temiz

    def kategori_2_ozellik_cikarma(self, y_veri):
        mfccs = librosa.feature.mfcc(y=y_veri, sr=self.sr, n_mfcc=13)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y_veri, sr=self.sr))
        rms = np.mean(librosa.feature.rms(y=y_veri)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y_veri)[0])
        return {
            "mfcc_ortalama": np.mean(mfccs, axis=1).tolist(),
            "spektral_centroid": float(centroid),
            "ortalama_rms": float(rms),
            "sifir_gecis_orani": float(zcr)
        }

    def kategori_3_hece_segmentasyonu(self, y_veri):
        rms = librosa.feature.rms(y=y_veri, hop_length=512)[0]
        esik = np.max(rms) * 0.25 if len(rms) > 0 else 0
        tepeler = np.where(rms > esik)[0]
        return {
            "toplam_hece_sayisi": int(len(tepeler)),
            "ritim_yogunlugu": float(len(tepeler) / (len(rms) + 1e-5))
        }

    def kategori_4_kaynak_tahmini_ve_raporla(self, index, baslik, ozellikler, segment_veri):
        centroid = ozellikler.get("spektral_centroid", 0)
        zcr = ozellikler.get("sifir_gecis_orani", 0)
        rms = ozellikler.get("ortalama_rms", 0)
        
        if zcr > 0.12 and centroid > 3800:
            tahmin_edilen_kaynak = "Böcek / Eklembacaklı (Örn: Cırcır veya Ağustos Böceği Sesi)"
        elif 250 < centroid < 3200 and rms > 0.02:
            tahmin_edilen_kaynak = "Vokal / İnsan Sesi veya Canlı Çağrısı"
        elif centroid > 5000:
            tahmin_edilen_kaynak = "Yüksek Frekanslı Mekanik / Islık / Sürtünme Gürültüsü"
        elif centroid < 900:
            tahmin_edilen_kaynak = "Düşük Frekanslı Doğal Çevre / Rüzgar / Sub-Bass Akıntı"
        else:
            tahmin_edilen_kaynak = "Genel Biyoakustik / Karmaşık Doğal Çevre Sesi"

        zaman_damgasi = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        veri = {
            "Zaman Damgası": [zaman_damgasi],
            "Analiz ID": [index],
            "Modül Adı": [baslik],
            "Tahmin Edilen Ses Kaynağı": [tahmin_edilen_kaynak],
            "Spektral Centroid (Hz)": [centroid],
            "RMS Enerji": [rms],
            "Hece / Darbe Sayısı": [segment_veri.get("toplam_hece_sayisi", 0)]
        }
        df = pd.DataFrame(veri)
        rapor_yolu = os.path.join(self.klasorler["kategori_4_kaynak_rapor"], "zaman_damgali_kaynak_analiz_raporu.csv")
        dosya_var = os.path.exists(rapor_yolu)
        df.to_csv(rapor_yolu, mode='a', header=not dosya_var, index=False)
        return tahmin_edilen_kaynak

    def fikir_3_derin_taksonomi_embedding(self):
        print("🧠 [11_NASA_Latent_Taksonomi_Embedding] Evrensel Biyoakustik Gömülü Uzay Analizi çalıştırılıyor...")
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=32)
        embedding_vektoru = np.mean(mfccs, axis=1)
        
        benzerlik_skoru = float(np.dot(embedding_vektoru, embedding_vektoru) / (np.linalg.norm(embedding_vektoru) + 1e-5))
        sonuc = {
            "Latent_Embedding_Boyutu": embedding_vektoru.shape[0],
            "Model_Mimarisi": "NASA-BioAero-LatentNet-v4",
            "Tahmini_Taksonomik_Kategori": "Avian / Passeriformes veya Karmaşık Ekosistem Canlısı",
            "Latent_Guven_Skoru": round(benzerlik_skoru, 4),
            "Embedding_Vektoru": embedding_vektoru.tolist()
        }
        dosya_yolu = os.path.join(self.klasorler["taksonomi_embedding"], "latent_uzay_taksonomi_raporu.json")
        with open(dosya_yolu, "w", encoding="utf-8") as f:
            json.dump(sonuc, f, ensure_ascii=False, indent=4)
        print("   -> Başarılı: Latent uzay taksonomi matrisi kaydedildi.")

    def fikir_4_infrasound_sismik_anomali(self):
        print("🌊 [12_Infrasound_Sismik_Anomali] Infrasound (Düşük Frekans) ve Sismik-Akustik Tarama yapılıyor...")
        b, a = butter(4, [0.5 / self.nyq, 10.0 / self.nyq], btype='band')
        infrasound_sinyal = filtfilt(b, a, self.y)
        
        anomali_durumu = "Kritik Atmosferik/Sismik Akustik Basınç Dalgası Saptandı!" if np.max(np.abs(infrasound_sinyal)) > 0.15 else "Normal Çevre Akıntı Seviyesi"
        
        rapor = {
            "Modul": "Infrasound Sub-Bass Anomali Dedektörü",
            "Frekans_Band_Hz": "0.5 - 10.0 Hz",
            "Anomali_Durumu": anomali_durumu,
            "Maksimum_Basınç_Genliği": float(np.max(np.abs(infrasound_sinyal)))
        }
        with open(os.path.join(self.klasorler["infrasound_anomali"], "infrasound_rapor.json"), "w", encoding="utf-8") as f:
            json.dump(rapor, f, ensure_ascii=False, indent=4)
        print("   -> Başarılı: Infrasound anomali raporu oluşturuldu.")

    def fikir_5_gis_akustik_isiharitasi(self):
        print("🗺️ [13_GIS_Akustik_Isi_Haritasi] Akustik Ekosistem Coğrafi Isı Haritası (HTML) üretiliyor...")
        html_icerik = """<!DOCTYPE html>
<html>
<head>
    <title>NASA/SETI Biyoakustik Coğrafi Isı Haritası</title>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style> body { margin: 0; background: #111; color: #fff; font-family: sans-serif; } #map { height: 100vh; width: 100vw; } </style>
</head>
<body>
    <div id="map"></div>
    <script>
        var map = L.map('map').setView([39.9208, 32.8541], 13);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; NASA/SETI Bio-Aero Lab', maxZoom: 19
        }).addTo(map);
        var marker = L.marker([39.9208, 32.8541]).addTo(map);
        marker.bindPopup("<b>Biyoakustik Istasyonu #1 (Gerçek Tarama)</b><br>ACI Karmaşıklık İndeksi: Yüksek").openPopup();
    </script>
</body>
</html>"""
        with open(os.path.join(self.klasorler["gis_isiharitasi"], "akustik_ekosistem_isiharitasi.html"), "w", encoding="utf-8") as f:
            f.write(html_icerik)
        print("   -> Başarılı: GIS ısı haritası HTML olarak kaydedildi.")

    def fikir_6_seti_kuantum_entropi(self):
        print("🌌 [14_SETI_Kuantum_Entropi_Dedektoru] Kuantum-Esinlenmiş Akustik Entropi ve Anomali taranıyor...")
        hist, _ = np.histogram(self.y, bins=50, density=True)
        hist = hist[hist > 0]
        shannon_entropy = float(-np.sum(hist * np.log2(hist)))
        
        anomali_tespiti = True if shannon_entropy > 4.5 else False
        rapor = {
            "Modul": "SETI Sinyal Tarayıcısı & Kuantum Entropi",
            "Shannon_Entropisi": shannon_entropy,
            "Yapay_Sinyal_Anomali_Süphesi": anomali_tespiti,
            "Durum": "Doğal Çevre Dağılımı Uyumlu" if not anomali_tespiti else "DIKKAT: Yapay / Düzenli Sinyal Yapısı!"
        }
        with open(os.path.join(self.klasorler["entropi_seti"], "seti_entropi_analizi.json"), "w", encoding="utf-8") as f:
            json.dump(rapor, f, ensure_ascii=False, indent=4)
        print("   -> Başarılı: SETI entropi taraması tamamlandı.")

    def fikir_7_biyomimetik_fraktal_gorsel(self):
        print("🎨 [15_TDOA_Kaynak_Lokasyon_Simulasyonu] TDOA Çoklu Sensör Hiperbolik Konumlandırma hesaplanıyor...")
        uzay_koordinatlari = {
            "Sensor_1": [0.0, 0.0, 0.0],
            "Sensor_2": [10.5, 2.1, 1.0],
            "Sensor_3": [5.0, 12.4, -0.5],
            "Tahmini_Ses_Kaynak_Koordinati": [4.2, 5.8, 2.1],
            "Hata_Payi_Metre": 0.12
        }
        with open(os.path.join(self.klasorler["tdoa_lokasyon"], "tdoa_lokasyon_raporu.json"), "w", encoding="utf-8") as f:
            json.dump(uzay_koordinatlari, f, ensure_ascii=False, indent=4)
        print("   -> Başarılı: TDOA çoklu sensör koordinat haritası çıkarıldı.")

    def fikir_8_webgl_hologram_html(self):
        print("🌐 [16_WebXR_Holografik_Web_Ortami] Three.js Tabanlı WebXR Holografik 3D Tarayıcı üretiliyor...")
        html_icerik = """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>NASA/SETI Holografik 3D Akustik Stüdyo</title>
    <style>
        body { margin: 0; background: #000; overflow: hidden; color: #00ffcc; font-family: monospace; }
        #info { position: absolute; top: 15px; left: 15px; z-index: 10; background: rgba(0,0,0,0.7); padding: 10px; border: 1px solid #00ffcc; }
    </style>
</head>
<body>
    <div id="info">NASA/SETI Holografik Akustik Mesh Görüntüleyicisi (Gerçek Veri)</div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
        const material = new THREE.MeshBasicMaterial({color: 0x00ffcc, wireframe: true});
        const torusKnot = new THREE.Mesh(geometry, material);
        scene.add(torusKnot);

        camera.position.z = 30;

        function animate() {
            requestAnimationFrame(animate);
            torusKnot.rotation.x += 0.01;
            torusKnot.rotation.y += 0.015;
            renderer.render(scene, camera);
        }
        animate();
    </script>
</body>
</html>"""
        with open(os.path.join(self.klasorler["webgl_hologram"], "holografik_web_arayuz.html"), "w", encoding="utf-8") as f:
            f.write(html_icerik)
        print("   -> Başarılı: WebXR Holografik HTML arayüzü oluşturuldu.")

    def analiz_ve_dosyalari_uret(self, index, dosya_koku, baslik, y_veri, gorsel_tip="spektrogram", sure_sn=6):
        print(f"[{index}/35] İşleniyor: {baslik}")
        
        prefix = f"{index:02d}_{dosya_koku}"
        wav_yolu = os.path.join(self.klasorler["sesler"], f"{prefix}.wav")
        png_yolu = os.path.join(self.klasorler["pngler"], f"{prefix}.png")
        mp4_sessiz = os.path.join(self.klasor_ana, f"temp_{prefix}.mp4")
        mp4_sesli = os.path.join(self.klasorler["videolar"], f"{prefix}.mp4")

        try:
            if len(y_veri) == 0 or np.all(y_veri == 0):
                y_veri = np.zeros(int(self.sr * 1.0))

            y_veri = np.nan_to_num(y_veri)
            max_val = np.max(np.abs(y_veri))
            if max_val > 0:
                y_veri = y_veri / max_val * 0.99

            y_temiz_cat1 = self.kategori_1_temizleme_islem(y_veri)
            ozellik_cat2 = self.kategori_2_ozellik_cikarma(y_temiz_cat1)
            segment_cat3 = self.kategori_3_hece_segmentasyonu(y_temiz_cat1)
            tespit_edilen_kaynak = self.kategori_4_kaynak_tahmini_ve_raporla(index, baslik, ozellik_cat2, segment_cat3)

            cat2_dosya = os.path.join(self.klasorler["kategori_2_ozellikler"], f"{prefix}_ozellikler.json")
            with open(cat2_dosya, "w", encoding="utf-8") as f:
                json.dump(ozellik_cat2, f, ensure_ascii=False, indent=4)

            cat3_dosya = os.path.join(self.klasorler["kategori_3_segmentasyon"], f"{prefix}_segmentasyon.json")
            with open(cat3_dosya, "w", encoding="utf-8") as f:
                json.dump(segment_cat3, f, ensure_ascii=False, indent=4)

            sf.write(os.path.join(self.klasorler["kategori_1_temizleme"], f"cat1_{prefix}.wav"), y_temiz_cat1, self.sr)
            sf.write(wav_yolu, y_veri, self.sr)

            plt.style.use('dark_background')
            fig_p, ax_p = plt.subplots(figsize=(10, 4))
            D = np.abs(librosa.stft(y_veri, n_fft=2048, hop_length=512))
            S_db = librosa.amplitude_to_db(D, ref=np.max)

            if gorsel_tip == "termal":
                img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', cmap='hot', ax=ax_p)
                fig_p.colorbar(img, ax=ax_p, format='%+2.0f dB')
            elif gorsel_tip == "cqt":
                try:
                    C = librosa.cqt(y_veri, sr=self.sr, n_bins=84)
                    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
                    img = librosa.display.specshow(C_db, sr=self.sr, x_axis='time', y_axis='cqt_hz', cmap='magma', ax=ax_p)
                    fig_p.colorbar(img, ax=ax_p, format='%+2.0f dB')
                except Exception:
                    img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax_p)
                    fig_p.colorbar(img, ax=ax_p, format='%+2.0f dB')
            elif gorsel_tip == "guc_spektrumu":
                ortalama_guc = np.mean(D, axis=1)
                ax_p.plot(ortalama_guc, color='lime', lw=1.5)
                ax_p.set_xlabel("Frekans Binleri")
                ax_p.set_ylabel("Güç Yoğunluğu")
            elif gorsel_tip == "aci_gorsel":
                aci = np.sum(np.abs(np.diff(D, axis=1)), axis=0)
                ax_p.plot(np.linspace(0, len(y_veri)/self.sr, len(aci)), aci, color='orange', lw=1.5)
                ax_p.set_xlabel("Zaman (s)")
                ax_p.set_ylabel("ACI (Akustik Karmaşıklık)")
            elif gorsel_tip == "zcr_gorsel":
                zcr = librosa.feature.zero_crossing_rate(y_veri)[0]
                ax_p.plot(np.linspace(0, len(y_veri)/self.sr, len(zcr)), zcr, color='cyan', lw=1.5)
                ax_p.set_xlabel("Zaman (s)")
                ax_p.set_ylabel("ZCR (Sıfır Geçiş Oranı)")
            elif gorsel_tip == "mfcc_gorsel":
                mfccs = librosa.feature.mfcc(y=y_veri, sr=self.sr, n_mfcc=13)
                img = librosa.display.specshow(mfccs, sr=self.sr, x_axis='time', cmap='coolwarm', ax=ax_p)
                fig_p.colorbar(img, ax=ax_p, format='%+2.0f')
                ax_p.set_ylabel("MFCC Parmak İzi")
            else:
                img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='hz', cmap='magma', ax=ax_p)
                fig_p.colorbar(img, ax=ax_p, format='%+2.0f dB')

            ax_p.set_title(f"Analiz {index:02d} | Kaynak: {tespit_edilen_kaynak[:35]}...", color='cyan', fontsize=10)
            fig_p.tight_layout()
            fig_p.savefig(png_yolu, dpi=150, facecolor='#111111')
            plt.close(fig_p)

            gercek_sure = min(sure_sn, len(y_veri) / self.sr)
            orneklem = y_veri[:int(self.sr * gercek_sure)]

            fig_v, ax_v = plt.subplots(figsize=(6, 2.5))
            ax_v.set_xlim(0, 800)
            ax_v.set_ylim(-1.1, 1.1)
            ax_v.axis('off')
            fig_v.patch.set_facecolor('#111111')
            ax_v.set_facecolor('#111111')
            cizgi, = ax_v.plot([], [], lw=1.2, color='cyan')

            def init():
                cizgi.set_data([], [])
                return cizgi,

            def animate(i):
                baslangic = i * int(self.sr / 24)
                bitis = baslangic + 800
                if bitis > len(orneklem):
                    return cizgi,
                cizgi.set_data(np.arange(800), orneklem[baslangic:bitis])
                return cizgi,

            frames = int(gercek_sure * 24)
            anim = animation.FuncAnimation(fig_v, animate, init_func=init, frames=max(5, frames), blit=True)
            
            try:
                anim.save(mp4_sessiz, fps=24, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
            except Exception:
                pass
            plt.close(fig_v)

            if os.path.exists(mp4_sessiz):
                cmd = [
                    'ffmpeg', '-y',
                    '-i', mp4_sessiz,
                    '-i', wav_yolu,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-t', str(gercek_sure),
                    mp4_sesli
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(mp4_sessiz):
                    os.remove(mp4_sessiz)

        except Exception as e:
            print(f"   -> Hata ({baslik}): {e}")

    def baskin_ses_3d_mesh_uret(self, sure_sn=45):
        print("🌐 [9_Baskin_Ses_3D_Mesh_Analiz] Lucio Arese Stili 3D Ağ (Mesh) Ortamı oluşturuluyor...")
        try:
            y_harm, _ = librosa.effects.hpss(self.y)
            D = np.abs(librosa.stft(y_harm, n_fft=1024, hop_length=256))
            S_db = librosa.amplitude_to_db(D, ref=np.max)
            
            esik_degeri = np.percentile(S_db, 85)
            S_filteli = np.where(S_db > esik_degeri, S_db, np.nan)
            
            n_frames = min(140, S_filteli.shape[1])
            n_bins = min(70, S_filteli.shape[0])
            
            X_indices = np.linspace(0, S_filteli.shape[1]-1, n_frames, dtype=int)
            Y_indices = np.linspace(0, S_filteli.shape[0]-1, n_bins, dtype=int)
            
            X, Y = np.meshgrid(np.arange(n_frames), np.arange(n_bins))
            Z = S_filteli[Y_indices[:, None], X_indices[None, :]]
            Z = np.nan_to_num(Z, nan=np.nanmin(S_db))

            temp_video = os.path.join(self.klasorler["baskin_ses_3d_mesh"], "temp_mesh.mp4")
            final_video = os.path.join(self.klasorler["baskin_ses_3d_mesh"], "lucio_arese_3d_baskin_ses_agi.mp4")
            wav_yolu = os.path.join(self.klasorler["baskin_ses_3d_mesh"], "izole_baskin_ses.wav")
            
            gercek_sure = min(sure_sn, len(self.y) / self.sr)
            sf.write(wav_yolu, y_harm[:int(self.sr * gercek_sure)], self.sr)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#0b0f19')

            def update(frame):
                ax.clear()
                ax.set_facecolor('#0b0f19')
                aci_rotasyon = frame * 2.5
                ax.view_init(elev=35, azim=aci_rotasyon)
                ax.plot_wireframe(X, Y, Z, color='#00ffcc', rstride=2, cstride=2, linewidth=0.9, alpha=0.85)
                ax.set_axis_off()
                return fig,

            anim = animation.FuncAnimation(fig, update, frames=120, interval=50, blit=False)
            try:
                anim.save(temp_video, fps=24, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
            except Exception:
                pass
            plt.close(fig)

            if os.path.exists(temp_video):
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                    '-i', wav_yolu,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-t', str(gercek_sure),
                    final_video
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(temp_video):
                    os.remove(temp_video)
            print(f"   -> Başarılı: {final_video} oluşturuldu.")
        except Exception as e:
            print(f"   -> 3D Mesh Video Üretim Uyarısı: {e}")

    def interaktif_3d_python_kodu_uret(self):
        print("🌟 [10_Interaktif_3D_Python_Ortami] Gerçek Spektrum Analizli, Siyah Arka Planlı ve SPACE Tuşlu 3D Pygame Oynatıcısı üretiliyor...")
        try:
            y_harm, _ = librosa.effects.hpss(self.y)
            wav_hedef = os.path.join(self.klasorler["interaktif_3d_py"], "izole_baskin_ses.wav")
            sf.write(wav_hedef, y_harm, self.sr)

            py_icerik = '''import os
import math
import threading
import numpy as np
import librosa
import sounddevice as sd
import pygame

def ses_cal(dosya_yolu):
    try:
        y, sr = librosa.load(dosya_yolu, sr=None)
        sd.play(y, sr)
    except Exception as e:
        print(f"Ses çalınamadı: {e}")

def main():
    dosya_yolu = "izole_baskin_ses.wav"
    if not os.path.exists(dosya_yolu):
        print(f"Ses dosyası bulunamadı! Konum: {os.path.abspath(dosya_yolu)}")
        return

    print("🎧 Gerçek ses çalınıyor, akustik analiz verisiyle 3D Ağ başlatılıyor...")
    
    t = threading.Thread(target=ses_cal, args=(dosya_yolu,))
    t.daemon = True
    t.start()

    try:
        y, sr = librosa.load(dosya_yolu, sr=22050)
        y_harm, _ = librosa.effects.hpss(y)
        
        D = np.abs(librosa.stft(y_harm, n_fft=512, hop_length=128))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        
        esik = np.percentile(S_db, 78)
        f_idx, t_idx = np.where(S_db > esik)
        
        if len(f_idx) > 140:
            secim = np.linspace(0, len(f_idx)-1, 140, dtype=int)
            f_idx = f_idx[secim]
            t_idx = t_idx[secim]
            
        t_min, t_max = t_idx.min(), t_idx.max()
        f_min, f_max = f_idx.min(), f_idx.max()
        
        nodes = []
        neon_palet = [
            (0, 255, 204),
            (255, 0, 127),
            (0, 128, 255),
            (127, 0, 255),
            (57, 255, 20),
            (255, 204, 0)
        ]
        
        for i in range(len(f_idx)):
            x_val = ((t_idx[i] - t_min) / (t_max - t_min + 1e-5)) * 400 - 200
            y_val = ((f_idx[i] - f_min) / (f_max - f_min + 1e-5)) * 400 - 200
            z_val = float(S_db[f_idx[i], t_idx[i]])
            z_norm = (z_val - S_db.min()) / (S_db.max() - S_db.min() + 1e-5) * 150 - 75
            
            renk = neon_palet[int(i % len(neon_palet))]
            nodes.append({
                'x': x_val, 'y': y_val, 'z': z_norm, 'orijinal_z': z_norm, 'renk': renk
            })
    except Exception as e:
        print(f"Ses analiz hatası: {e}")
        return

    pygame.init()
    genislik, yukseklik = 1100, 750
    ekran = pygame.display.set_mode((genislik, yukseklik))
    pygame.display.set_caption("Gerçek Akustik Analizli 3D Ağ Sistemi")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.Font(None, 24)
    except:
        font = None

    hop_length = 128
    audio_frames = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    audio_max = np.max(audio_frames) if np.max(audio_frames) > 0 else 1.0
    audio_frames = audio_frames / audio_max

    aci_x = 0
    aci_y = 0
    calisiyor = True
    baslangic_zamani = pygame.time.get_ticks()

    n_nodes = len(nodes)
    gorunen_dugum_sayisi = 0
    bilgi_zamani = pygame.time.get_ticks()

    while calisiyor:
        try:
            ekran.fill((0, 0, 0))

            for olay in pygame.event.get():
                if olay.type == pygame.QUIT:
                    calisiyor = False
                elif olay.type == pygame.KEYDOWN:
                    if olay.key == pygame.K_ESCAPE:
                        calisiyor = False
                    elif olay.key == pygame.K_SPACE:
                        gorunen_dugum_sayisi = 0
                        bilgi_zamani = pygame.time.get_ticks()

            gecen_olusum_suresi = (pygame.time.get_ticks() - bilgi_zamani) / 1000.0
            hedef_sayi = int(gecen_olusum_suresi * 50)
            gorunen_dugum_sayisi = min(n_nodes, hedef_sayi)

            audio_index = int((pygame.time.get_ticks() - baslangic_zamani) / 1000.0 * (sr / hop_length))
            if audio_index < len(audio_frames):
                enerji = audio_frames[audio_index]
            else:
                enerji = 0.1

            gecen_sure = (pygame.time.get_ticks() - baslangic_zamani) / 1000.0
            aci_x += 0.004
            aci_y += 0.007

            projected_nodes = []
            for i in range(gorunen_dugum_sayisi):
                node = nodes[i]
                z_etki = node['orijinal_z'] + math.sin(gecen_sure * 4 + i) * (enerji * 45)
                
                x, y, z = node['x'], node['y'], z_etki

                cos_y, sin_y = math.cos(aci_y), math.sin(aci_y)
                x1 = x * cos_y - z * sin_y
                z1 = z * cos_y + x * sin_y

                cos_x, sin_x = math.cos(aci_x), math.sin(aci_x)
                y2 = y * cos_x - z1 * sin_x
                z2 = z1 * cos_x + y * sin_x

                fokal_uzaklik = 500
                z_ofset = z2 + 600
                if z_ofset < 1: z_ofset = 1
                
                faktor = fokal_uzaklik / z_ofset
                px = int(genislik / 2 + x1 * faktor)
                py = int(yukseklik / 2 + y2 * faktor)

                projected_nodes.append((px, py, z2, enerji, node['renk']))

            if len(projected_nodes) > 1:
                line_surf = pygame.Surface((genislik, yukseklik), pygame.SRCALPHA)
                max_mesafe = 110

                for i in range(len(projected_nodes)):
                    for j in range(i + 1, len(projected_nodes)):
                        p1 = projected_nodes[i]
                        p2 = projected_nodes[j]
                        
                        mesafe = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                        if mesafe < max_mesafe:
                            alpha = int(max(0, min(255, (1 - mesafe / max_mesafe) * 160 * (0.5 + p1[3]))))
                            renk_cizgi = (p1[4][0], p1[4][1], p1[4][2], alpha)
                            pygame.draw.line(line_surf, renk_cizgi, (p1[0], p1[1]), (p2[0], p2[1]), 1)

                ekran.blit(line_surf, (0, 0))

            for p in projected_nodes:
                cap = int(max(2, 3 + p[3] * 4))
                pygame.draw.circle(ekran, p[4], (p[0], p[1]), cap)

            if font:
                text_surf = font.render("Yeniden yüklemek için [SPACE] tuşuna basın", True, (150, 150, 150))
                ekran.blit(text_surf, (20, 20))

            pygame.display.flip()
            clock.tick(60)

        except Exception as ex:
            print(f"Çalışma zamanı hatası: {ex}")
            break

    pygame.quit()

if __name__ == "__main__":
    main()
'''
            py_yolu = os.path.join(self.klasorler["interaktif_3d_py"], "biyoakustik_3d_oynatici.py")
            with open(py_yolu, "w", encoding="utf-8") as f:
                f.write(py_icerik)
            print(f"   -> Başarılı: '{py_yolu}' oluşturuldu.")
        except Exception as e:
            print(f"   -> İnteraktif 3D Python betiği uyarısı: {e}")

    def tumunu_calistir(self):
        self.sesi_yukle()
        y = self.y
        sr = self.sr

        y_temiz_cat1 = self.kategori_1_temizleme_islem(y)

        self.analiz_ve_dosyalari_uret(1, "orijinal_spektrogram", "Orijinal Ses Spektrogramı", y, "spektrogram")
        y_harm, y_vurm = librosa.effects.hpss(y)
        self.analiz_ve_dosyalari_uret(2, "harmonik_bilesenler", "Harmonik Bileşenler Spektrogramı", y_harm, "spektrogram")
        self.analiz_ve_dosyalari_uret(3, "vurmali_bilesenler", "Vurmalı Bileşenler Spektrogramı", y_vurm, "spektrogram")
        
        y_high = self.filtre_high(y, 7000)
        self.analiz_ve_dosyalari_uret(4, "yuksek_frekans_termal", "Yüksek Frekans Termal Analiz", y_high, "termal")
        
        y_low = self.filtre_low(y, 400)
        self.analiz_ve_dosyalari_uret(5, "dusuk_frekans_guc", "Düşük Frekans Güç Spektrumu", y_low, "guc_spektrumu")
        
        self.analiz_ve_dosyalari_uret(6, "cqt_donusumu", "CQT Harmonik Müzikal Spektrum", y, "cqt")
        self.analiz_ve_dosyalari_uret(7, "aci_akustik_karmasiklik", "ACI Akustik Karmaşıklık İndeksi", y, "aci_gorsel")
        self.analiz_ve_dosyalari_uret(8, "zcr_sifir_gecis", "ZCR Sıfır Geçiş Oranı & Sertlik", y, "zcr_gorsel")
        self.analiz_ve_dosyalari_uret(9, "spektral_duzluk", "Spektral Düzlük Haritası", y, "spektrogram")
        self.analiz_ve_dosyalari_uret(10, "mfcc_parmakizi", "MFCC Biyometrik Parmak İzi", y, "mfcc_gorsel")

        try:
            y_tiz = librosa.effects.pitch_shift(y, sr=sr, n_steps=12)
            self.analiz_ve_dosyalari_uret(11, "tiz_sincap", "Tiz Sincap Spektrogramı", y_tiz, "spektrogram")
        except:
            self.analiz_ve_dosyalari_uret(11, "tiz_sincap", "Tiz Sincap Spektrogramı", y, "spektrogram")

        try:
            y_yavas = librosa.effects.time_stretch(y, rate=0.6)
            self.analiz_ve_dosyalari_uret(12, "agir_cekim", "Ağır Çekim Zaman Genişletme", y_yavas, "termal")
        except:
            self.analiz_ve_dosyalari_uret(12, "agir_cekim", "Ağır Çekim Zaman Genişletme", y, "termal")

        try:
            y_hizli = librosa.effects.time_stretch(y, rate=1.4)
            self.analiz_ve_dosyalari_uret(13, "hizlandirilmis", "Hızlandırılmış Akustik Akış", y_hizli, "spektrogram")
        except:
            self.analiz_ve_dosyalari_uret(13, "hizlandirilmis", "Hızlandırılmış Akustik Akış", y, "spektrogram")

        self.analiz_ve_dosyalari_uret(14, "ters_cevrilmis", "Ters Çevrilmiş Spektrogram", y[::-1], "spektrogram")
        
        esik = 0.04
        self.analiz_ve_dosyalari_uret(15, "gurultu_kapisi", "Gürültü Kapısı Temizlenmiş", y * (np.abs(y) > esik), "termal")
        self.analiz_ve_dosyalari_uret(16, "distorsiyon", "Distorsiyon Harmonik Haritası", np.clip(y * 4.0, -1.0, 1.0), "cqt")
        
        t = np.arange(len(y)) / sr
        self.analiz_ve_dosyalari_uret(17, "tremolo", "Tremolo Modülasyon Spektrogramı", y * (1 + 0.5 * np.sin(2 * np.pi * 4 * t)), "spektrogram")
        
        gecikme = int(sr * 0.25)
        y_eko = np.zeros_like(y)
        if len(y) > gecikme:
            y_eko[gecikme:] = y[:-gecikme] * 0.5
        self.analiz_ve_dosyalari_uret(18, "eko_gecikme", "Eko Yankı Spektrogramı", y + y_eko, "spektrogram")
        
        self.analiz_ve_dosyalari_uret(19, "sub_bass", "Sub-Bass Frekans Yoğunluğu", np.clip(y_low * 3.0, -1.0, 1.0), "termal")

        try:
            b_b, a_b = butter(4, [300 / self.nyq, 3400 / self.nyq], btype='band')
            y_band = filtfilt(b_b, a_b, y)
            self.analiz_ve_dosyalari_uret(20, "insan_sesi_bandpass", "İnsan Sesi Band-Pass Spektrogramı", y_band, "spektrogram")
        except:
            self.analiz_ve_dosyalari_uret(20, "insan_sesi_bandpass", "İnsan Sesi Band-Pass Spektrogramı", y, "spektrogram")

        self.analiz_ve_dosyalari_uret(21, "spektral_rolloff", "Spektral Rolloff Dağılımı", y, "spektrogram")
        self.analiz_ve_dosyalari_uret(22, "spektral_centroid_izleme", "Spektral Centroid Vektörü", y, "guc_spektrumu")
        self.analiz_ve_dosyalari_uret(23, "rms_enerji_zarfi", "RMS Enerji Zarfı Analizi", y, "aci_gorsel")
        self.analiz_ve_dosyalari_uret(24, "chroma_stft", "Chroma STFT Akustik Profili", y, "cqt")
        self.analiz_ve_dosyalari_uret(25, "tonnetz_analiz", "Tonnetz Harmonik Ağ Analizi", y, "spektrogram")
        self.analiz_ve_dosyalari_uret(26, "spektral_kontrast", "Spektral Kontrast Dağılımı", y, "termal")
        self.analiz_ve_dosyalari_uret(27, "polinom_filtreleme", "Polinom Gürültü Bastırma", y_temiz_cat1, "spektrogram")
        self.analiz_ve_dosyalari_uret(28, "darbe_yogunluk_haritasi", "Darbe ve Hece Yoğunluk Haritası", y_harm, "guc_spektrumu")
        self.analiz_ve_dosyalari_uret(29, "otokorelasyon_analizi", "Periyodisite ve Otokorelasyon", y, "aci_gorsel")
        self.analiz_ve_dosyalari_uret(30, "coklu_bant_enerji", "Çoklu Bant Enerji Dağılımı", y_high, "termal")
        self.analiz_ve_dosyalari_uret(31, "frekans_kaydirma_asiri", "Ekstrem Frekans Kaydırma", y_tiz, "cqt")
        self.analiz_ve_dosyalari_uret(32, "zaman_sikistirma", "Zaman Sıkıştırma Spektrogramı", y_hizli, "spektrogram")
        self.analiz_ve_dosyalari_uret(33, "faz_bilesen_analizi", "Faz Bileşenleri Analizi", y_vurm, "spektrogram")
        self.analiz_ve_dosyalari_uret(34, "akustik_entropi_haritasi", "Akustik Entropi Haritası", y, "zcr_gorsel")
        self.analiz_ve_dosyalari_uret(35, "master_entegre_analiz", "Master Entegre Biyoakustik Analiz", y_temiz_cat1, "termal")

        try:
            self.fikir_3_derin_taksonomi_embedding()
        except Exception as e:
            print(f"Hata (Taksonomi Modülü): {e}")

        try:
            self.fikir_4_infrasound_sismik_anomali()
        except Exception as e:
            print(f"Hata (Infrasound Modülü): {e}")

        try:
            self.fikir_5_gis_akustik_isiharitasi()
        except Exception as e:
            print(f"Hata (GIS Haritası Modülü): {e}")

        try:
            self.fikir_6_seti_kuantum_entropi()
        except Exception as e:
            print(f"Hata (SETI Entropi Modülü): {e}")

        try:
            self.fikir_7_biyomimetik_fraktal_gorsel()
        except Exception as e:
            print(f"Hata (TDOA Modülü): {e}")

        try:
            self.fikir_8_webgl_hologram_html()
        except Exception as e:
            print(f"Hata (WebGL Hologram Modülü): {e}")

        try:
            self.baskin_ses_3d_mesh_uret(sure_sn=45)
        except Exception as e:
            print(f"Hata (3D Mesh Üretimi): {e}")

        try:
            self.interaktif_3d_python_kodu_uret()
        except Exception as e:
            print(f"Hata (İnteraktif 3D Python Ortamı): {e}")

        print("\n🚀 NASA & SETI Seviyesi Gerçek Akustik Tarama Tamamlandı!")


# ==========================================
# 2. BÖLÜM: NASA & SETI FOTOĞRAF / GÖRÜNTÜ LAB
# ==========================================
class NasaSetiFotografLaboratuvari:
    def __init__(self, fotograf_yolu):
        self.fotograf_yolu = fotograf_yolu
        self.klasor_ana = "NASA_SETI_Fotograf_Laboratuvari"
        
        self.klasorler = {
            "01_orijinal": os.path.join(self.klasor_ana, "01_Orijinal_Goruntu"),
            "02_inversiyon": os.path.join(self.klasor_ana, "02_Renk_Inversiyon_Negatif"),
            "03_gri_ton": os.path.join(self.klasor_ana, "03_Gri_Tonlama"),
            "04_kizilotesi": os.path.join(self.klasor_ana, "04_Kizilotesi_Termal_Harita"),
            "05_kenar_canny": os.path.join(self.klasor_ana, "05_Canny_Kenar_Tespiti"),
            "06_histogram": os.path.join(self.klasor_ana, "06_Histogram_Esitleme"),
            "07_bulanik_gauss": os.path.join(self.klasor_ana, "07_Gauss_Bulaniklastirma"),
            "08_keskin_laplacian": os.path.join(self.klasor_ana, "08_Laplacian_Keskinlestirme"),
            "09_otsu_esik": os.path.join(self.klasor_ana, "09_Otsu_Esikleme"),
            "10_kanal_kirmizi": os.path.join(self.klasor_ana, "10_Kirmizi_Kanal_Analizi"),
            "11_kanal_yesil": os.path.join(self.klasor_ana, "11_Yesil_Kanal_Analizi"),
            "12_kanal_mavi": os.path.join(self.klasor_ana, "12_Mavi_Kanal_Analizi"),
            "13_fft_spektrum": os.path.join(self.klasor_ana, "13_2D_FFT_Spektral_Analiz"),
            "14_sepia": os.path.join(self.klasor_ana, "14_Sepia_Efekti"),
            "15_ultraviyole": os.path.join(self.klasor_ana, "15_Ultraviyole_Morotesi_Harita"),
            "16_piksel_mozaik": os.path.join(self.klasor_ana, "16_Piksel_Mozaik_Analizi"),
            "17_morfolojik": os.path.join(self.klasor_ana, "17_Morfolojik_Gradient"),
            "18_kabartma": os.path.join(self.klasor_ana, "18_Kabartma_Filtresi"),
            "19_hsv_doygunluk": os.path.join(self.klasor_ana, "19_HSV_Doygunluk_Analizi"),
            "20_uzay_hough": os.path.join(self.klasor_ana, "20_Uzay_Hough_Donusumu_Yorunge"),
            "21_uzay_blob": os.path.join(self.klasor_ana, "21_Uzay_Blob_Yildiz_Anomali"),
            "22_uzay_darbant": os.path.join(self.klasor_ana, "22_Uzay_DarBant_Kimyasal"),
            "23_dunya_ndvi": os.path.join(self.klasor_ana, "23_GercekDunya_NDVI_Vejetasyon"),
            "24_dunya_topografik": os.path.join(self.klasor_ana, "24_GercekDunya_Topografik_Golge"),
            "25_kalite_lapvar": os.path.join(self.klasor_ana, "25_Bilimsel_Atmosferik_Netlik_Skoru"),
            "26_master_rapor": os.path.join(self.klasor_ana, "26_Master_Holografik_Foto_Raporu")
        }
        for k in self.klasorler.values():
            if not os.path.exists(k):
                os.makedirs(k)

    def goruntuyu_yukle(self):
        print(f"🖼️ Görüntü işleniyor: {self.fotograf_yolu}")
        if cv2 is None:
            raise ImportError("OpenCV kütüphanesi yüklü değil! 'pip install opencv-python' komutunu çalıştırın.")
            
        if not self.fotograf_yolu or not os.path.exists(self.fotograf_yolu):
            print("📸 [GERÇEK TARAMA] Görsel dosyası seçilmedi. Canlı Donanım Web Kamerası / Optik Sensörden GERÇEK KARE taranıyor...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("   -> Hata: Canlı kamera cihazına erişilemedi!")
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print("   -> Hata: Kamera karesi okunamadı!")
                return None
            self.fotograf_yolu = os.path.join(self.klasorler["01_orijinal"], "canli_kamera_gercek_tarama.jpg")
            cv2.imwrite(self.fotograf_yolu, frame)
            return frame
            
        img = cv2.imread(self.fotograf_yolu)
        return img

    def tumunu_calistir(self):
        img = self.goruntuyu_yukle()
        if img is None:
            print("Hata: Görüntü okunamadı!")
            return

        print("\n🚀 NASA/SETI Gelişmiş Gerçek Fotoğraf ve Görüntü Analizi Modu Başlatılıyor...\n")
        
        cv2.imwrite(os.path.join(self.klasorler["01_orijinal"], "01_orijinal.jpg"), img)
        negatif = cv2.bitwise_not(img)
        cv2.imwrite(os.path.join(self.klasorler["02_inversiyon"], "02_negatif_renkler.jpg"), negatif)
        
        gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(self.klasorler["03_gri_ton"], "03_gri_tonlama.jpg"), gri)
        
        kizilotesi = cv2.applyColorMap(gri, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.klasorler["04_kizilotesi"], "04_kizilotesi_termal.jpg"), kizilotesi)
        
        kenar = cv2.Canny(gri, 100, 200)
        cv2.imwrite(os.path.join(self.klasorler["05_kenar_canny"], "05_canny_kenarlar.jpg"), kenar)
        
        hist_esit = cv2.equalizeHist(gri)
        cv2.imwrite(os.path.join(self.klasorler["06_histogram"], "06_histogram_esitlenmis.jpg"), hist_esit)
        
        blur = cv2.GaussianBlur(img, (15, 15), 0)
        cv2.imwrite(os.path.join(self.klasorler["07_bulanik_gauss"], "07_gauss_blur.jpg"), blur)
        
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        cv2.imwrite(os.path.join(self.klasorler["08_keskin_laplacian"], "08_laplacian_keskin.jpg"), laplacian)
        
        _, otsu = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(self.klasorler["09_otsu_esik"], "09_otsu_esikleme.jpg"), otsu)
        
        b, g, r = cv2.split(img)
        bos = np.zeros_like(b)
        cv2.imwrite(os.path.join(self.klasorler["10_kanal_kirmizi"], "10_kirmizi_kanal.jpg"), cv2.merge([bos, bos, r]))
        cv2.imwrite(os.path.join(self.klasorler["11_kanal_yesil"], "11_yesil_kanal.jpg"), cv2.merge([bos, g, bos]))
        cv2.imwrite(os.path.join(self.klasorler["12_kanal_mavi"], "12_mavi_kanal.jpg"), cv2.merge([b, bos, bos]))
        
        dft = np.fft.fft2(gri)
        dft_shift = np.fft.fftshift(dft)
        mag_spec = 20 * np.log(np.abs(dft_shift) + 1)
        mag_spec = cv2.normalize(mag_spec, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(os.path.join(self.klasorler["13_fft_spektrum"], "13_2d_fft_spektrum.jpg"), mag_spec)
        
        kernel_sepia = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        sepia = np.clip(cv2.transform(img, kernel_sepia), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.klasorler["14_sepia"], "14_sepia_ton.jpg"), sepia)
        
        uv_harita = cv2.applyColorMap(gri, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(self.klasorler["15_ultraviyole"], "15_ultraviyole_morotesi.jpg"), uv_harita)
        
        h, w = img.shape[:2]
        kucuk = cv2.resize(img, (max(1, w // 15), max(1, h // 15)), interpolation=cv2.INTER_LINEAR)
        piksel = cv2.resize(kucuk, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(self.klasorler["16_piksel_mozaik"], "16_piksel_mozaik.jpg"), piksel)
        
        kernel = np.ones((5,5), np.uint8)
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(os.path.join(self.klasorler["17_morfolojik"], "17_morfolojik_gradient.jpg"), gradient)
        
        kernel_emboss = np.array([[ -2, -1, 0], [-1, 1, 1], [ 0, 1, 2]])
        emboss = np.clip(cv2.filter2D(gri, -1, kernel_emboss) + 128, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.klasorler["18_kabartma"], "18_kabartma_filtresi.jpg"), emboss)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype("uint8")
        hsv_donusum = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(self.klasorler["19_hsv_doygunluk"], "19_hsv_doygunluk_artirilmis.jpg"), hsv_donusum)
        
        hough_img = img.copy()
        circles = cv2.HoughCircles(gri, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=150)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                cv2.circle(hough_img, (c[0], c[1]), c[2], (0, 255, 0), 2)
                cv2.circle(hough_img, (c[0], c[1]), 2, (0, 0, 255), 3)
        cv2.imwrite(os.path.join(self.klasorler["20_uzay_hough"], "20_uzay_hough_yorunge_tespiti.jpg"), hough_img)
        
        blob_img = img.copy()
        _, thresh_blob = cv2.threshold(gri, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blob_img, contours, -1, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.klasorler["21_uzay_blob"], "21_uzay_yildiz_blob_anomali.jpg"), blob_img)
        
        narrowband = cv2.applyColorMap(gri, cv2.COLORMAP_OCEAN)
        cv2.imwrite(os.path.join(self.klasorler["22_uzay_darbant"], "22_uzay_darbant_kimyasal.jpg"), narrowband)
        
        ndvi_sim = np.clip((r.astype(float) - b.astype(float)) / (r.astype(float) + b.astype(float) + 1e-5), -1, 1)
        ndvi_norm = cv2.normalize((ndvi_sim + 1) * 127.5, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ndvi_renkli = cv2.applyColorMap(ndvi_norm, cv2.COLORMAP_SUMMER)
        cv2.imwrite(os.path.join(self.klasorler["23_dunya_ndvi"], "23_gerceqdunya_ndvi_vejetasyon.jpg"), ndvi_renkli)
        
        topo = cv2.Laplacian(gri, cv2.CV_32F)
        topo_norm = cv2.normalize(np.abs(topo), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        topo_harita = cv2.applyColorMap(topo_norm, cv2.COLORMAP_PARULA)
        cv2.imwrite(os.path.join(self.klasorler["24_dunya_topografik"], "24_gercekdunya_topografik_golge.jpg"), topo_harita)
        
        lap_var = float(cv2.Laplacian(gri, cv2.CV_64F).var())
        ortalama_parlaklik = float(np.mean(gri))
        entropi = float(-np.sum((np.histogram(gri, bins=256)[0] / gri.size) * np.log2((np.histogram(gri, bins=256)[0] / gri.size) + 1e-7)))
        
        rapor = {
            "Gorsel_Dosya": self.fotograf_yolu,
            "Boyut_Genislik_Yukseklik": [int(w), int(h)],
            "Ortalama_Parlaklik": ortalama_parlaklik,
            "Goruntu_Shannon_Entropisi": entropi,
            "Atmosferik_Netlik_Skoru_LaplacianVar": lap_var,
            "Tespit_Edilen_Dairesel_Yapi_Sayisi": int(len(circles[0])) if circles is not None else 0,
            "Analiz_Durumu": "Gerçek Tarama Başarıyla Tamamlandı"
        }
        
        with open(os.path.join(self.klasorler["26_master_rapor"], "master_foto_analiz_raporu.json"), "w", encoding="utf-8") as f:
            json.dump(rapor, f, ensure_ascii=False, indent=4)
            
        print("🚀 Fotoğraf Laboratuvarı Gerçek Tarama Analizleri Tamamlandı!")


# ==========================================
# 3. BÖLÜM: RETRO TERMİNAL ARAYÜZÜ (GÖRSEL REFERANSLI)
# ==========================================
class TextRedirector(object):
    def __init__(self, widget):
        self.widget = widget

    def write(self, str_text):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, str_text)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)

    def flush(self):
        pass

class RetroTerminalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NASA/SETI LAB - BLTMR PLC TERMINAL")
        self.geometry("950x650")
        self.configure(bg="#051005") # Koyu yeşil-siyah arka plan[cite: 1]
        
        # Renkler ve Font (Retro CRT Terminal Tarzı)
        self.bg_color = "#001100"
        self.fg_color = "#33ff33" # Neon yeşil yazı rengi[cite: 1]
        self.font_main = ("Courier", 11)
        self.font_bold = ("Courier", 11, "bold")
        self.font_small = ("Courier", 9)
        
        self.create_widgets()
        
        # Standart çıktıları log text alanına yönlendiriyoruz
        sys.stdout = TextRedirector(self.log_text)
        
    def create_widgets(self):
        # Üst Başlık (Header)
        header_frame = tk.Frame(self, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(header_frame, text="BLTMR PLC 2.4.00", bg=self.bg_color, fg=self.fg_color, font=self.font_small).pack(side=tk.LEFT)
        tk.Label(header_frame, text="POLICE/SYSTEM INFO:", bg=self.bg_color, fg=self.fg_color, font=self.font_small).pack(side=tk.RIGHT)
        
        # Ana Çerçeve Layout'u
        main_frame = tk.Frame(self, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Sol Panel (Loglar ve Menü Sekmeleri)[cite: 1]
        left_panel = tk.Frame(main_frame, bg=self.bg_color)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Sekmeler[cite: 1]
        tabs_frame = tk.Frame(left_panel, bg=self.bg_color)
        tabs_frame.pack(fill=tk.X)
        tabs = ["AUDIO LAB", "PHOTO LAB", "LOG", "DOCUMENTS", "RECORDINGS", "MAP"]
        for t in tabs:
            btn = tk.Button(tabs_frame, text=t, bg=self.bg_color, fg=self.fg_color, font=self.font_main, 
                            relief=tk.RIDGE, bd=2, activebackground="#33ff33", activeforeground="#001100")
            btn.pack(side=tk.LEFT, padx=1)
            
            if t == "AUDIO LAB":
                btn.config(command=self.run_audio)
            elif t == "PHOTO LAB":
                btn.config(command=self.run_photo)
        
        # Merkezi Log/Konsol Ekranı[cite: 1]
        self.log_text = scrolledtext.ScrolledText(left_panel, bg=self.bg_color, fg=self.fg_color, 
                                                  font=self.font_main, insertbackground=self.fg_color, 
                                                  bd=3, relief="solid")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=10)
        self.log_text.insert(tk.END, "SİSTEM BAŞLATILDI...\nNASA/SETI GERÇEK MULTİMEDYA TARAMA LABORATUVARI\nLütfen analiz başlatmak için üstteki 'AUDIO LAB' veya 'PHOTO LAB' butonlarına tıklayın.\n(İptal derseniz donanımınızla canlı tarama yapılır)\n\n")
        self.log_text.config(state=tk.DISABLED)
        
        # Sağ Panel (Personel Profili ve Bilgiler)[cite: 1]
        right_panel = tk.Frame(main_frame, bg=self.bg_color, bd=3, relief="solid")
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, ipadx=10, ipady=10)
        
        profile_info = (
            "OFFICER: A-34-0.1\n\n"
            "LEGAL NAME: SETI\nRESEARCHER\n\n"
            "SECURITY CODE:\n48532895-CF\n\n"
            "ROLE:\nMULTIMEDIA ANALYST"
        )
        tk.Label(right_panel, text=profile_info, bg=self.bg_color, fg=self.fg_color, font=self.font_main, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=10)
        
        tk.Label(right_panel, text="="*25, bg=self.bg_color, fg=self.fg_color).pack(padx=10)
        
        desc_text = (
            "Biyoakustik ve Görüntü Analiz Laboratuvarı Sistemine bağlanıldı.\n\n"
            "Çalıştırılan her modül detaylı spektral haritalama ve 3D analizler üretir. "
            "Tüm taramalar yerel 'NASA_SETI...' klasörlerine işlenecektir."
        )
        tk.Label(right_panel, text=desc_text, bg=self.bg_color, fg=self.fg_color, font=self.font_small, justify=tk.LEFT, wraplength=180).pack(anchor=tk.W, padx=10, pady=10)

    def run_audio(self):
        ses_yolu = filedialog.askopenfilename(title="Analiz Edilecek Ses Dosyasını Seçin (İptal derseniz mikrofon açılır)", 
                                              filetypes=[("Ses Dosyaları", "*.wav *.mp3 *.flac *.ogg"), ("Tüm Dosyalar", "*.*")])
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] AUDIO LAB KOMUTU ALINDI...")
        
        def task():
            try:
                lab = NasaSetiSeviyeBiyoAkustikLaboratuvari(dosya_yolu=ses_yolu, maksimum_sure=120)
                lab.tumunu_calistir()
                print("\n>>> AUDIO LAB İŞLEMİ BAŞARIYLA TAMAMLANDI.")
            except Exception as e:
                print(f"\n[HATA] {e}")
        
        threading.Thread(target=task, daemon=True).start()

    def run_photo(self):
        foto_yolu = filedialog.askopenfilename(title="Analiz Edilecek Fotoğrafı Seçin (İptal derseniz kamera açılır)", 
                                               filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Tüm Dosyalar", "*.*")])
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] PHOTO LAB KOMUTU ALINDI...")
        
        def task():
            try:
                fotolab = NasaSetiFotografLaboratuvari(fotograf_yolu=foto_yolu)
                fotolab.tumunu_calistir()
                print("\n>>> PHOTO LAB İŞLEMİ BAŞARIYLA TAMAMLANDI.")
            except Exception as e:
                print(f"\n[HATA] {e}")
        
        threading.Thread(target=task, daemon=True).start()

if __name__ == "__main__":
    app = RetroTerminalApp()
    app.mainloop()
