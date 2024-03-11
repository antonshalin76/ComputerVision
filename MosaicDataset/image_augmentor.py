from PIL import Image
import albumentations as A
import numpy as np
import concurrent.futures
from tqdm import tqdm

# Класс аугментации пар картинка-аннотации на базе библиотеки albumentations
class ImageAugmentor:
    def __init__(self):
        self.augmentations = A.Compose([
                A.RandomCropFromBorders(p=0.33, crop_left=0.05, crop_right=0.05, crop_top=0.05, crop_bottom=0.05),
                A.Rotate(p=0.33, limit=7, interpolation=0, border_mode=4),
                A.ShiftScaleRotate(p=0.33, shift_limit_x=0.05, shift_limit_y=0.05, scale_limit=0.1, rotate_limit=0, interpolation=0, border_mode=4),
                A.HorizontalFlip(p=0.4),
                A.RGBShift(p=0.33, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
                A.RandomBrightnessContrast(p=0.33),
                A.CLAHE(p=0.33, clip_limit=(1, 4), tile_grid_size=(8, 8)),
                A.GaussNoise(p=0.33, var_limit=(10.0, 50.0), per_channel=True, mean=0.0),
#                A.ElasticTransform(p=0.25, alpha=1, sigma=20, alpha_affine=20), # Эластичные трансформации для деформации изображения
#                A.OpticalDistortion(p=0.25, distort_limit=0.02, shift_limit=0.02), # Оптическое искажение
                A.CoarseDropout(p=0.33, max_holes=8, max_height=8, max_width=8) # Создает случайные пропущенные пиксели
        ], 
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
    # Применение аугментаций к изображению, bbox
    def _apply_augmentations(self, image, bboxes, class_labels):
        try:
            augmented = self.augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
            return augmented['image'], augmented['bboxes']
        except Exception as e:
            return image, bboxes

    # Метод для коррекции BBox-ов после аугментации
    def _correct_bboxes(self, class_labels, transformed_bboxes):
        corrected_bboxes = []
        for bbox, label in zip(transformed_bboxes, class_labels):
            # Переставляем метку класса на первое место в списке
            corrected_bbox = [label] + list(bbox)
            corrected_bboxes.append(corrected_bbox)
        return corrected_bboxes

    # Функция для аугментации одного изображения
    def _augment_single_image(self, img_with_bbox):
        img, bboxes = img_with_bbox
        img_array = np.array(img)  # Конвертируем PIL.Image в numpy array

        # Извлечение class_labels из bboxes
        class_labels = [bbox[0] for bbox in bboxes]
        bboxes_without_labels = [bbox[1:] for bbox in bboxes]

        # Применение аугментаций
        transformed_img, transformed_bboxes = self._apply_augmentations(
            img_array, bboxes_without_labels, class_labels)
              
        # Конвертируем обратно в PIL.Image после аугментаций
        transformed_img = Image.fromarray(transformed_img)

        # Коррекции bboxes
        corrected_bboxes = self._correct_bboxes(class_labels, transformed_bboxes)
        
        return (transformed_img, corrected_bboxes)

    # Основная функция аугментации всех доступных пар картинка-аннотации
    def augment_images(self, images_with_bboxes):
        if not self.augmentations:  # Проверяем, есть ли аугментации
            return images_with_bboxes  # Возвращаем изображения без изменений, если аугментаций нет

        # Использование многопоточности для аугментации изображений
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._augment_single_image, images_with_bboxes), total=len(images_with_bboxes), desc="Аугментация исходных фото", unit=" images"))

        return results