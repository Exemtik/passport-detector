# Обработка паспортных данных при помощи YOLOv5 и OCR(Tesseract)

Этот проект демонстрирует полный конвейер для обнаружения и извлечения полей паспорта из изображений с использованием обнаружения объектов (YOLOv5) и OCR (Tesseract). Конвейер реализован на Python и предназначен для запуска из командной строки как консольного приложения.

---

## Project Structure

```
passport_ocr_pipeline/
├── data/
│   ├── original/
│   │   └── ...
│   ├── test/
│   │   └── passport_2.png
├── ├── out_with_boxes.png
├── labeled_data/
│   ├── images/
│   └── result.json
├── augmented_data/
│   ├── images/
│   └── result.json
├── yolo_dataset/
│   └── ... (преобразованные аннотации YOLO)
├── yolo_dataset_split/
│   └── ... (Разделенные файлы на train and val)
├── yolov5/
│   └── ... (клонирован из ultralytics/yolov5)
├── JSON2YOLO/
│   └── ... (клонирован из ultralytics/JSON2YOLO)
├── augment.py
├── data.yaml
├── detect_and_extract.py
├── draw.py
├── requirements.txt
├── results.json
├── README.md
```

---

## Инструкции по настройке

Клонировать репозиторий:
```bash
git clone https://github.com/yourusername/passport-ocr.git
cd passport-ocr
```

### 1. Настройка среды

```bash
git clone https://github.com/yourusername/passport-ocr.git
cd passport-ocr
```

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Клонировать YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

# Установить JSON2YOLO
git clone https://github.com/ultralytics/JSON2YOLO
cd JSON2YOLO
pip install -r requirements.txt
```

Установите Tesseract OCR:
- Следуйте инструкциям для вашей ОС: https://github.com/tesseract-ocr/tesseract
- Для поддержки русского языка загрузите языковой пакет

---

## Пошаговый конвейер

### Шаг 1: Маркировка (вручную)

Используйте [Label Studio](https://labelstud.io/) для маркировки изображений и экспорта проекта в формат COCO. Сохраните вывод в каталоге `labeled_data/`:

```
labeled_data/
├── images/
├── result.json
```

### Шаг 2: Дополнение

Запустите скрипт `augment.py`, чтобы применить Albumentations и сохранить дополненные изображения и метки COCO.

```bash
python augment.py
```

Это создаст папку `augmented_data/` с дополненными изображениями и аннотациями.

### Шаг 3: Конвертируйте формат COCO в YOLO

Используйте JSON2YOLO для преобразования аннотаций:

```bash
cd JSON2YOLO
python convert.py --json_dir path/to/coco/annotations --save_dir path/to/yolo/labels
```

### Шаг 4: Обучение YOLOv5

Используем `data.yaml` файл, который содержит:

```yaml
train: ../yolo_dataset/images/train
val: ../yolo_dataset/images/val
nc: 9
names: ["Date of birth", "Gender", "Last name", "Name", "Passport number", "Passport series", "Photo", "Place of birth", "Surname"]
```


Обучите модель, используя:

```bash
cd yolov5
python train.py --img 640 --batch 8 --epochs 100 --data ../data.yaml --weights yolov5s.pt --name passport_yolo --cache
```

### Шаг 5: Обнаружение

Выполнить вывод на тестовом изображении:

```bash
python detect.py --weights runs/train/passport_yolo/weights/best.pt --source ../data/test/passport_2.png --save-txt --save-conf
```

### Шаг 6: Извлечение полей с помощью OCR

Запустите скрипт извлечения:

```bash
python detect_and_extract.py
```

Для распознавания текста каждой интересующей области используется `pytesseract`.

---

## Описания скриптов

### `augment.py`

* Загружает аннотации COCO
* Применяет 10 дополнений к изображению
* Выводит измененные изображения 512x512 и новый COCO JSON

### `detect_and_extract.py`

* Считывает метки YOLO
* Извлекает рамки с максимальной достоверностью
* Выполняет OCR для каждой области интереса (исключая фотографию)
* Поворачивает определенные поля для лучших результатов OCR

### `draw.py`

* Извлекает рамки с максимальной достоверностью
* Отрисовывает прозрачные прямоугольники с классами, имеющими наивысшую точность
---

## Пример результата

Результат, полученный при помощи YOLO5
![Результат обучения YOLO5](yolov5\runs\detect\exp2\passport_2.png)

Результат, сгенерированный при помощи `draw.py`
![Заново полученное изображение](data\output_with_boxes.png)
```
{
  "Date of birth": "16.12.1984",
  "Gender": "МУЖ",
  "Last name": "НИСЕРХОВИЧ",
  "Name": "ГАМЛЕТ",
  "Passport number": "9000007",
  "Passport series": "11 04",
  "Place of birth": "ле.ГОР. АРХАНГЕЛЬСК",
  "Surname": "БЕСТРЕВ."
}
```

---

## Лицензия

This project is under the MIT License.
