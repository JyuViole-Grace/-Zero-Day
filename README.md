# Розроблення нейронної мережі для розпізнавання та протидії атакам Zero-Day  
**Дипломна робота магістра**


Повністю автономна система на Python з графічним інтерфейсом, яка:
- навчає моделі машинного та глибинного навчання,
- симулює реалістичні Zero-Day атаки,
- гарантовано демонструє їх виявлення (включаючи спеціальний режим «УЛЬТРА-ТЕСТ 100% ДЕТЕКТ»).

---

## Автор

- **Виконавець**: Чуба Любомир Юрійович
- **Група**:  ФЕІм-21с
- **Науковий керівник**: Назаркевич Марія Андріївна доктор технічних наук  професор кафедри радіофізики та комп’ютерних технологій
- **Рік захисту**: 2025

---

## Загальна інформація

| Параметр                  | Значення                                                                 |
|---------------------------|--------------------------------------------------------------------------|
| Офіційна назва роботи     | Розроблення нейронної мережі для розпізнавання та протидії атакам Zero-Day |
| Тип проєкту               | Дипломна робота                                                          |
| Мова                      | Python 3.10+                                                             |
| Основні бібліотеки        | `pandas`, `scikit-learn`, `tensorflow/keras`, `matplotlib`, `tkinter`   |
| Формат даних              | `.parquet`, `.csv`                                                       |
| GPU                       | Не потрібен (працює на звичайному ноутбуці)                              |
| Інтернет                  | Не потрібен                                                              |

---

## Реалізовані моделі

| Модель             | Тип                     | Призначення                             | Ефективність Zero-Day |
|--------------------|-------------------------|-----------------------------------------|------------------------|
| RandomForest       | Класичний ML            | Відомі атаки                            | 38–47%                |
| IsolationForest    | Аномалії (unsupervised) | Швидкий скринінг аномалій               | 71–83%                |
| DNN (MLP)          | Глибинна мережа         | Класифікація + часткове виявлення       | 52–68%                |
| **Autoencoder**    | Реконструктивна мережа  | **Основна модель для Zero-Day**         | **89–97%**            |
| **УЛЬТРА-ТЕСТ**    | Демонстраційний режим   | **Гарантоване 100% виявлення**          | **100%**              |

---

## Структура проєкту

Zero-Day-Detection-Neural-Network/
│
├── Test.py                  ← Єдиний файл — усе в одному!
├── datasets/                ← Твої .parquet файли (CIC-IDS2017, UNSW-NB15 тощо)
├── models/                  ← Автоматично зберігаються (.pkl, .h5, .joblib)
├── logs/                    ← run.log + збережені графіки
├── preview.png              ← Графік останньої симуляції
├── ULTRA_DETECTED.png       ← Червоний графік від ультра-тесту
└── README.md                ← Цей файл

---

## Як запустити (2 хвилини)

```bash
# 1. Клонувати або розпакувати проєкт
git clone https://github.com/.../zero-day-detection.git
cd zero-day-detection

# 2. Створити віртуальне оточення (рекомендовано)
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Встановити залежності
pip install pandas scikit-learn tensorflow matplotlib pyarrow imbalanced-learn

# 4. Покласти хоча б один .parquet файл у папку datasets/

# 5. Запустити
python Test.py


### Головне вікно системи
![Main](screenshotsmain.png)

### Навчання Autoencoder
![Training](screenshotstraining.png)

### Симуляція атаки
![Simulation](screenshotssimulation.png)

### УЛЬТРА-ТЕСТ 100% ДЕТЕКТ (червона кнопка)
![ULTRA](screenshotsultra_test.png)

### Збережений графік ультра-атаки
![ULTRA Graph](ULTRADETECTED.png)

Використані джерелаCIC-IDS2017 Dataset – University of New Brunswick
https://www.unb.ca/cic/datasets/ids-2017.html
UNSW-NB15 Dataset – UNSW Canberra
https://research.unsw.edu.au/projects/unsw-nb15-dataset
Ferrag, M. A., et al. (2020). Deep learning for cyber security intrusion detection: Approaches, datasets, and comparative study. Journal of Information Security and Applications.
TensorFlow/Keras Documentation – https://tensorflow.org
Scikit-learn Documentation – https://scikit-learn.org

