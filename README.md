# Dating Uygulaması Öneri Sistemi

## 📖 Proje Hakkında
Bu proje, bir dating uygulamasında kullanıcıların profillerine dayalı öneriler sunmayı amaçlayan bir öneri sistemi geliştirmektedir. Kullanıcıların ilgi alanları, yaş, boy, aktivite düzeyi gibi özellikleri analiz edilerek, en uygun eşleşmelerin sağlanması hedeflenmektedir. Proje, temel veri analizi teknikleri ve kişiselleştirilmiş öneri sistemlerinin nasıl kullanılabileceğini göstermektedir.

### Amaç
Projenin amacı, kullanıcılar arasında daha iyi eşleşmeleri sağlamak için bir öneri sistemi oluşturmaktır. Kullanıcıların tercihleri ve özellikleri dikkate alınarak, en uygun eşleşmelerin belirlenmesi sağlanır.

### Kullanılan Teknolojiler
- **Python**: Projenin ana programlama dili.
- **Pandas**: Veri manipülasyonu ve analizi için kullanılır.
- **NumPy**: Sayısal hesaplamalar için kullanılır.
- **Plotly**: Görselleştirme için kullanılır.
- **Scikit-learn**: Veri ön işleme ve modelleme için kullanılır.

### Proje Yapısı
Proje, aşağıdaki adımları içermektedir:
1. **Veri Yükleme ve Ön İşleme**: Veri setinin yüklenmesi, eksik veri kontrolü ve temel istatistiklerin hesaplanması.
2. **Veri Görselleştirme**: Kullanıcıların yaş dağılımı, ilgi alanları analizi ve eğitim seviyesi ile yaş ilişkisini görselleştirme.
3. **Yeni Özellikler Ekleme**: Kullanıcıların ilgi alanı sayısı ve aktivite skoru gibi yeni özelliklerin eklenmesi.
4. **Öneri Algoritması**: Kullanıcı profilleri arasındaki benzerlikleri hesaplayarak öneri skoru belirleme.
5. **Performans Değerlendirme**: Öneri sisteminin performansının ölçülmesi ve görselleştirilmesi.

### Kullanım
Proje, Python ortamında çalıştırılabilir. Kullanıcıdan alınan girişlere göre öneriler sunmak için aşağıdaki adımlar izlenmelidir:
1. Gerekli kütüphanelerin yüklenmesi.
2. Veri setinin yüklenmesi ve analizi.
3. Öneri algoritmasının işleyişinin test edilmesi.

### Proje Sonuçları
- Kullanıcıların çoğunluğunun 25-35 yaş aralığında olduğu belirlenmiştir.
- En popüler ilgi alanları arasında spor, müzik ve seyahat bulunmaktadır.
- Geliştirilen eşleşme algoritması, %85'in üzerinde eşleşme skoru üretebilmektedir.
- Kullanıcıların ilişki hedefleri ve çocuk tercihleri konusunda yüksek hassasiyet gösterilmektedir.

### Geliştirme Önerileri
- Makine öğrenmesi modellerinin entegrasyonu.
- Gerçek zamanlı etkileşim verilerinin eklenmesi.
- Kullanıcı geri bildirimlerine dayalı öğrenme mekanizması.
- Coğrafi konum bazlı filtreleme özellikleri.

## 🚧 Sorunlar
Proje ile ilgili herhangi bir sorun veya geliştirme önerisi için iletişime geçebilirsiniz. Gerekli veri setinin ve bağımlılıkların doğru bir şekilde yüklendiğinden emin olunmalıdır.

## 📄 Lisans
Bu proje MIT Lisansı altında lisanslanmıştır.

## 📦 Model Kaydetme
Eğitilen scaler modeli `scaler.pkl` dosyası olarak kaydedilmiştir. Bu, modelin daha sonraki kullanımlar için yeniden yüklenmesini sağlar.
