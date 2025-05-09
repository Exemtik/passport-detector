import os
import re
import json
import cv2
import pytesseract
from scipy import ndimage

# Executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CLASS_NAMES = [
    "Date of birth", "Gender", "Last name", "Name", "Passport number",
    "Passport series", "Photo", "Place of birth", "Surname"
]


def load_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return image, lines


def get_best_detections(label_lines, class_names):
    class_confidences = {name: 0.0 for name in class_names}
    best_detections = {}

    for line in label_lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        conf = float(parts[5])
        class_name = class_names[class_id]

        if conf > class_confidences[class_name]:
            class_confidences[class_name] = conf
            best_detections[class_name] = parts

    return best_detections


def extract_text_from_image(image, detections, class_names, output_dir):
    h, w, _ = image.shape
    results = {name: '' for name in class_names if name != "Photo"}

    os.makedirs(output_dir, exist_ok=True)

    for i, (class_name, parts) in enumerate(detections.items()):
        class_id = int(parts[0])
        xc, yc, bw, bh, conf = map(float, parts[1:])

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        roi = image[y1:y2, x1:x2]

        if class_name == "Photo":
            continue

        text = pytesseract.image_to_string(roi, lang='rus').strip()
        clean_text = re.sub(r'[^а-яА-Яa-zA-Z0-9.,\- ]', '', text)
        results[class_name] = clean_text

        if class_name in ["Passport number", "Passport series"]:
            roi = ndimage.rotate(roi, 45)

        roi_filename = os.path.join(output_dir, f"{i}_{class_name}.jpg")
        cv2.imwrite(roi_filename, roi)

    return results


def save_results(results, output_path='results.json'):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    image_path = 'data/test/passport_2.png'
    label_path = 'yolov5/runs/detect/exp2/labels/passport_2.txt'
    output_dir = "rois"

    image, label_lines = load_image_and_labels(image_path, label_path)
    best_detections = get_best_detections(label_lines, CLASS_NAMES)
    results = extract_text_from_image(image, best_detections, CLASS_NAMES, output_dir)
    save_results(results)


if __name__ == '__main__':
    main()
