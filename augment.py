import os
import json
import uuid
import cv2
import albumentations as A
from tqdm import tqdm


def load_coco_annotation(src_dir):
    with open(os.path.join(src_dir, 'result.json'), 'r', encoding='utf-8') as f:
        coco = json.load(f)
    return coco['images'], coco['annotations'], coco['categories']


def build_annotation_map(annotations):
    image_to_annotations = {}
    for ann in annotations:
        image_to_annotations.setdefault(ann['image_id'], []).append(ann)
    return image_to_annotations


def get_transform():
    return A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.MotionBlur(p=0.2),
        A.Resize(512, 512),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


def augment_dataset(src_dir, dest_dir, num_aug):
    os.makedirs(os.path.join(dest_dir, 'images'), exist_ok=True)

    images, annotations, categories = load_coco_annotation(src_dir)
    image_to_annotations = build_annotation_map(annotations)
    transform = get_transform()

    new_images = []
    new_annotations = []
    ann_id = 1
    img_id = 1

    for image in tqdm(images, desc="Augmenting images"):
        image_path = os.path.join(src_dir, 'images', image['file_name'])
        img = cv2.imread(image_path)
        if img is None:
            continue

        bboxes = []
        category_ids = []
        anns = image_to_annotations.get(image['id'], [])

        for ann in anns:
            bboxes.append(ann['bbox'])
            category_ids.append(ann['category_id'])

        for _ in range(num_aug):
            aug = transform(image=img, bboxes=bboxes, category_ids=category_ids)
            aug_img = aug['image']
            aug_bboxes = aug['bboxes']
            aug_cats = aug['category_ids']

            filename = f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(os.path.join(dest_dir, 'images', filename), aug_img)

            new_images.append({
                'id': img_id,
                'file_name': filename,
                'height': 512,
                'width': 512
            })

            for bbox, cat_id in zip(aug_bboxes, aug_cats):
                new_annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 0
                })
                ann_id += 1
            img_id += 1

    aug_coco = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }

    with open(os.path.join(dest_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(aug_coco, f, ensure_ascii=False, indent=2)


def main():
    src_dir = 'labeled_data'
    dest_dir = 'augmented_data'
    number_augmentation = 10

    augment_dataset(src_dir, dest_dir, number_augmentation)


if __name__ == '__main__':
    main()
