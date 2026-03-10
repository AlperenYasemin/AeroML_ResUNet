# BÖLÜM 4: Kurulum, API Entegrasyonu ve Gelecek Geliştirmeler (Deployment, API Integration & Future Work)

## 4.1. Kurulum ve Çalıştırma Ortamı (Deployment Environment)

Sistem, yerel veya bulut tabanlı bir sunucuda bağımsız bir süreç (microservice) olarak çalıştırılmak üzere tasarlanmıştır.

* **Gereksinimler:** Python 3.12, PyTorch (CUDA destekli), FastAPI, Uvicorn.
* **Başlatma (Cold Start):** Sunucu terminal üzerinden `uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload` komutu ile ayağa kaldırılır. Başlatma anında diskteki ağırlık dosyası RAM/VRAM'e çekilerek kullanıma hazır bekletilir.
* **Konteynerizasyon (Docker):** Üretim ortamında (production) bağımlılık çakışmalarını önlemek için sistemin bir `Dockerfile` ile paketlenmesi zorunludur. Hesaplama hızını korumak için NVIDIA Container Toolkit kullanılarak GPU donanımının konteyner içine doğrudan geçirilmesi (passthrough) gerekmektedir.

## 4.2. API İletişim Protokolü (API Integration)

İstemci (Web/Mobil) ile sunucu arasındaki iletişim, HTTP REST protokolü üzerinden sağlanır. Sistem durumsuz (stateless) çalışır; sunucu geçmiş tahminlerin kaydını tutmaz.

* **Uç Nokta (Endpoint):** `POST /predict`
* **Veri Yükü (Payload):** İstemci, sınır koşullarını (`ux_in`, `uy_in`) JSON formatında gönderir.
* **Yanıt (Response):** API, modelden çıkan ham tensör matrislerini sunucu tarafında işler, renkli ısı haritalarına (Heatmap) dönüştürür ve Base64 formatında metin olarak kodlayıp JSON gövdesi içinde geri döndürür. Bu tasarım kararı, istemci tarafındaki (özellikle zayıf mobil cihazlardaki) render yükünü tamamen ortadan kaldırır.

## 4.3. Üretime Geçişteki Darboğazlar (Production Bottlenecks)

Sistemin endüstriyel standartlarda gerçek zamanlı bir uygulamaya dönüşmesi için şu darboğazların aşılması gerekmektedir:

1. **Model Optimizasyonu:** Model ağırlıkları şu an ham PyTorch (`.pth`) formatındadır. Çıkarım (inference) hızını maksimize etmek ve bellek tüketimini azaltmak için model grafiğinin ONNX (Open Neural Network Exchange) veya TensorRT motorlarına derlenmesi (export) şarttır.
2. **Eşzamanlılık (Concurrency):** FastAPI asenkron istekleri karşılayabilse de, tek bir GPU'nun aynı anda işleyebileceği matris çarpımı donanımsal olarak sınırlıdır. Yüksek trafik altında (örneğin aynı anda 50 kullanıcı parametre değiştirdiğinde) sistemin çökmemesi için Celery veya RabbitMQ gibi bir görev kuyruğu (message broker) entegre edilmelidir.

## 4.4. Gelecek Geliştirmeler ve Araştırma Vizyonu (Future Work)

Bilimsel Makine Öğrenmesi (Scientific ML) standartları göz önüne alındığında, projenin mevcut sürümü bir kavram kanıtıdır (Proof of Concept). Sonraki aşamalarda ele alınması gereken kritik konular şunlardır:

1. **Fizik Bilgili Sinir Ağları (PINNs):** Sistemin en büyük zayıflığı, kaybı hesaplarken yalnızca Ortalama Kare Hata (MSE) kullanmasıdır. Model, Navier-Stokes denklemlerinden (kütle ve momentum korunumu) tamamen habersizdir. Çıktıların fiziksel olarak geçerli olabilmesi için kayıp fonksiyonuna kısmi diferansiyel denklemlerin (PDE) bir ceza terimi olarak eklenmesi gerekmektedir.
2. **Uzamsal Artefaktların Giderilmesi:** Çıktılarda gözlemlenen ızgara (checkerboard) dokusunu yok etmek için, Decoder katmanındaki standart `ConvTranspose2d` operasyonları *Bilinear Upsampling* yöntemleri ile değiştirilmelidir.
3. **Karmaşık Akış Dinamikleri:** 2 boyutlu daimi akış (steady-state) simülasyonlarından, zaman faktörünü de içeren kararsız (transient) ve 3 boyutlu akışların tahminine (Spatio-temporal modeling) geçiş yapılmalıdır.
