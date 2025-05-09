import os
import cv2
import numpy as np

CLASS_NAMES = {
    0: "Date of birth",
    1: "Gender",
    2: "Last name",
    3: "Name",
    4: "Passport number",
    5: "Passport series",
    6: "Photo",
    7: "Place of birth",
    8: "Surname"
}


def get_color(class_id):
    np.random.seed(class_id)
    return tuple(int(c) for c in np.random.randint(0, 255, 3))


def read_image_and_labels(image_path, label_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    return image, lines


def get_best_detections(lines):
    best_detections = {}

    for line in lines:
        parts = list(map(float, line.strip().split()))
        cls_id, x_c, y_c, bw, bh, conf = parts
        cls_id = int(cls_id)

        if cls_id not in best_detections or conf > best_detections[cls_id][-1]:
            best_detections[cls_id] = (x_c, y_c, bw, bh, conf)

    return best_detections


def draw_boxes(image, detections, class_names, alpha=0.4):
    overlay = image.copy()
    h_img, w_img = image.shape[:2]

    for cls_id, (x_c, y_c, bw, bh, conf) in detections.items():
        x1 = int((x_c - bw / 2) * w_img)
        y1 = int((y_c - bh / 2) * h_img)
        x2 = int((x_c + bw / 2) * w_img)
        y2 = int((y_c + bh / 2) * h_img)

        color = get_color(cls_id)
        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 2), font, 0.5, (255, 255, 255), 1)

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    return image


def main():
    image_path = "data/test/passport_2.png"
    label_path = "yolov5/runs/detect/exp2/labels/passport_2.txt"
    output_path = "data/output_with_boxes.png"

    image, lines = read_image_and_labels(image_path, label_path)
    best_detections = get_best_detections(lines)
    annotated_image = draw_boxes(image, best_detections, CLASS_NAMES)
    cv2.imwrite(output_path, annotated_image)


if __name__ == '__main__':
    main()
