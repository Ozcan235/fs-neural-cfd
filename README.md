# Formula Student Neural CFD

**English** | [Türkçe](#türkçe)

A **surrogate model** that instantly predicts the aerodynamic performance of a Formula Student front wing in ground effect (lift coefficient **Cl** and drag coefficient **Cd**) without running a CFD simulation.

> **Status:** In development. MVP target: February 2027.

## Why?

CFD simulations are reliable but slow: evaluating a single wing setting can take minutes to hours. For teams with limited CFD experience or computing power, testing hundreds of settings is not practical.

This project trains a machine learning model on a dataset generated with OpenFOAM and validated against published experimental data. Once trained, the model predicts Cl and Cd for a new wing setting in milliseconds.

## How it works

1. **Simulation and validation:** 2D front wing simulations in OpenFOAM, compared with experimental data and checked for mesh independence.
2. **Automated pipeline:** Meshing (Gmsh), solving (OpenFOAM) and post-processing (Python) run with a single command.
3. **Dataset:** 300–500 simulations generated with Sobol / Latin hypercube sampling, published as an open dataset.
4. **Models:** Linear regression, gradient boosting and a neural network (MLP), with uncertainty estimates.
5. **Demo:** A browser-based tool built with FastAPI and React.

| Input | Output |
|---|---|
| Angle of attack | Cl |
| Ride height | Cd |
| Flap angle | Prediction uncertainty |

**Tech stack:** Python, OpenFOAM, Gmsh, ParaView, NumPy, pandas, scikit-learn, PyTorch, FastAPI, React

## Scope and limitations

- **2D, single airfoil.** 3D effects such as endplates and tip vortices are not modeled.
- **Isolated wing.** Interaction with the wheels and the rest of the car is not modeled.
- **Limited to the training range.** Predictions outside the trained parameter range are not reliable; the demo warns the user in that case.
- **An early-stage screening tool.** It helps find promising settings quickly; it does not replace a team's own CFD analysis.
- **The model cannot be more accurate than its CFD data.** This is why validating the simulations against experiments is the foundation of the project.

## Roadmap

- [x] Development environment
- [ ] Aerodynamics and CFD fundamentals
- [ ] Wing simulation in OpenFOAM and validation against experiments
- [ ] Automated data generation pipeline
- [ ] Dataset generation and release
- [ ] Machine learning models and comparison
- [ ] Web demo and release

## References

- NASA Glenn Research Center, *Beginner's Guide to Aeronautics*
- Zerihan and Zhang, experimental studies on front wings in ground effect (validation data)

## License

To be decided.

## Contact

I'm open to feedback, collaboration, and internship or job opportunities.

**Özcan Ersin** — Management Information Systems, Necmettin Erbakan University

- LinkedIn: [Özcan Ersin](https://www.linkedin.com/in/%C3%B6zcan-ersin-816b5b2a6/)
- Email: [ty_ozcan@hotmail.com](mailto:ty_ozcan@hotmail.com)
- GitHub: [@Ozcan235](https://github.com/Ozcan235)

---

## Türkçe

Formula Student ön kanadının yer etkisindeki aerodinamik performansını (kaldırma katsayısı **Cl** ve sürükleme katsayısı **Cd**), CFD simülasyonu çalıştırmadan anında tahmin eden bir **vekil model (surrogate model)**.

> **Durum:** Geliştirme aşamasında. MVP hedefi: Şubat 2027.

### Neden?

CFD simülasyonları güvenilir sonuç verir ama yavaştır: tek bir kanat ayarını değerlendirmek dakikalar ya da saatler sürebilir. CFD bilgisi veya hesaplama gücü kısıtlı olan takımlar için yüzlerce farklı ayarı denemek pratik değildir.

Bu proje, OpenFOAM ile üretilip yayımlanmış deney verisiyle doğrulanmış bir veri seti üzerinde bir makine öğrenmesi modeli eğitir. Eğitilen model, yeni bir kanat ayarı için Cl ve Cd değerlerini milisaniyeler içinde tahmin eder.

### Nasıl çalışıyor?

1. **Simülasyon ve doğrulama:** OpenFOAM ile 2B ön kanat simülasyonu; sonuçlar deney verisiyle karşılaştırılır ve ağ bağımsızlığı test edilir.
2. **Otomatik veri hattı:** Ağ oluşturma (Gmsh), çözüm (OpenFOAM) ve sonuç okuma (Python) tek komutla çalışır.
3. **Veri seti:** Sobol / Latin hiperküp örneklemesiyle 300–500 simülasyon üretilir ve açık veri seti olarak yayımlanır.
4. **Model:** Doğrusal regresyon, gradyan artırma ve sinir ağı (MLP) karşılaştırılır; belirsizlik tahmini yapılır.
5. **Demo:** FastAPI ve React ile tarayıcıdan kullanılabilen bir arayüz.

| Girdi | Çıktı |
|---|---|
| Hücum açısı | Cl |
| Yerden yükseklik | Cd |
| Flap açısı | Tahminin belirsizliği |

### Kapsam ve sınırlar

- **2B ve tek kanat profili.** Uç plakaları ve kanat ucu girdapları gibi 3B etkiler hesaba katılmaz.
- **İzole kanat.** Tekerlekler ve aracın geri kalanıyla etkileşim modellenmez.
- **Eğitim aralığıyla sınırlı.** Model, eğitildiği parametre aralığı dışında güvenilir değildir; demo bu durumda uyarı verir.
- **Ön tarama aracıdır.** Umut verici ayarları hızlıca bulmaya yardımcı olur; takımların kendi CFD analizlerinin yerini tutmaz.
- **Model, CFD verisinden daha doğru olamaz.** Bu yüzden simülasyonların deneyle doğrulanması projenin temelidir.

### İletişim

Geri bildirime, iş birliğine ve staj ya da iş fırsatlarına açığım. İletişim bilgileri için yukarıdaki [Contact](#contact) bölümüne bakabilirsiniz.
