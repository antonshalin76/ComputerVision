import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Класс разделения исходных данных на три набора обучения модели (train/val/test)
class SplitSubset:
    def __init__(self, image_bbox_pairs, split_ratio, tolerance=0.05):
        self.image_bbox_pairs = image_bbox_pairs
        self.split_ratio = split_ratio
        self.tolerance = tolerance  # Допустимое отклонение
        self.class_entities = self._count_class_entities()

    # Считаем общее количество сущностей каждого класса
    def _count_class_entities(self):
        class_entities = {}
        for _, bboxes in self.image_bbox_pairs:
            for bbox in bboxes:
                class_id = bbox[0]
                class_entities[class_id] = class_entities.get(class_id, 0) + 1
        return class_entities

    # Распределение сущностей по наборам согласно split_ratio
    def _distribute_entities(self):
        distributed_entities = {i: {} for i in range(len(self.split_ratio))}
        for class_id, total in self.class_entities.items():
            for i, ratio in enumerate(self.split_ratio):
                entities_count = int(total * ratio)
                tolerance_count = int(self.tolerance * total)
                distributed_entities[i][class_id] = max(entities_count - tolerance_count, 0)
        return distributed_entities
    
    # Выбор поднабора, куда лучше положить взятую картинку
    def _select_subset(self, class_counts, distributed_entities, remaining_entities):
        best_subset = -1
        min_difference = float('inf')
        for i, subset_entities in distributed_entities.items():
            difference = 0
            for class_id, count in class_counts.items():
                difference += abs((subset_entities[class_id] - count) - remaining_entities[class_id])
            if difference < min_difference:
                min_difference = difference
                best_subset = i
        return best_subset 

    # Основная функция стратифицированного распределения сущностей по поднаборам
    def split(self):
        subsets = [[] for _ in self.split_ratio]
        remaining_entities = self.class_entities.copy()
        distributed_entities = self._distribute_entities()
        random.shuffle(self.image_bbox_pairs)

        for img, bboxes in tqdm(self.image_bbox_pairs, desc="Splitting data", unit=" pair"):
            class_counts = {class_id: 0 for class_id in self.class_entities}
            for bbox in bboxes:
                class_id = bbox[0]
                class_counts[class_id] += 1
            best_subset = self._select_subset(class_counts, distributed_entities, remaining_entities)
            if best_subset != -1:
                subsets[best_subset].append((img, bboxes))
                for class_id, count in class_counts.items():
                    remaining_entities[class_id] -= count
                    distributed_entities[best_subset][class_id] -= count
            else:
                min_size = min(len(subset) for subset in subsets)
                for subset in subsets:
                    if len(subset) == min_size:
                        subset.append((img, bboxes))
                        break

        return subsets
    
    # Функция для сохранения разделенных наборов данных в соответствующие папки
    def save_splits(self, directory):
        # Разделяем данные с помощью метода split
        train_set, valid_set, test_set = self.split()

        # Словарь для хранения соответствий между наборами данных и их папками
        split_sets = {"train": train_set, "valid": valid_set, "test": test_set}
        for subset_name, subset in split_sets.items():
            # Создаем папки для изображений и аннотаций
            subset_img_dir = os.path.join(directory, subset_name, 'images')
            subset_labels_dir = os.path.join(directory, subset_name, 'labels')
            os.makedirs(subset_img_dir, exist_ok=True)
            os.makedirs(subset_labels_dir, exist_ok=True)

            # Сохранение изображений и аннотаций
            for idx, (image, bboxes) in enumerate(subset):
                image_path = os.path.join(subset_img_dir, f'image_{idx}.jpg')
                image.save(image_path) # Сохраняем изображение

                bbox_path = os.path.join(subset_labels_dir, f'image_{idx}.txt')
                with open(bbox_path, 'w') as file:
                    for bbox in bboxes:
                        file.write(' '.join(map(str, bbox)) + '\n') # Сохраняем аннотации
                        
    # Функция для загрузки данных из папок в наборы train, val и test
    def load_sets_from_folders(self, folders):
        loaded_sets = {}
        for subset_name, folder in folders.items():
            images_dir = os.path.join(folder, 'images')
            labels_dir = os.path.join(folder, 'labels')

            image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            label_paths = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('.txt')]

            image_label_pairs = []
            for img_path, lbl_path in zip(image_paths, label_paths):
                img = Image.open(img_path)
                bboxes = mosaic_creator._load_bboxes(lbl_path)  # Использование метода _load_bboxes из MosaicCreator
                image_label_pairs.append((img, bboxes))

            loaded_sets[subset_name] = image_label_pairs

        return loaded_sets['train'], loaded_sets['valid'], loaded_sets['test']                      

    # Функция для проверки и обновления наборов данных
    def check_and_update(self, src_directory, dst_directory):
        total_dst_pairs = 0
        for subset in ['train', 'valid', 'test']:
            subset_dir = os.path.join(dst_directory, subset)
            dst_pairs = self.find_image_label_pairs(subset_dir)
            total_dst_pairs += len(dst_pairs)

        return total_src_pairs == total_dst_pairs    
    
    # Функция для подсчета количества аннотаций по каждому классу в представленном поднаборе данных
    def count_annotations_by_class(self, dataset):
        class_counts = {}
        for _, bboxes in dataset:
            for bbox in bboxes:
                class_id = int(bbox[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        # Сортировка словаря по ключам
        sorted_class_counts = dict(sorted(class_counts.items()))
        return sorted_class_counts  