from PIL import Image
import os
import shutil
from tqdm import tqdm
import concurrent.futures

# Вспомогательная функция для удаления директории
def delete_directory(path: str):
    # Удаляет директорию по заданному пути, если она существует
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f'[ERROR] Error deleting a directory: `{path}`: {e.strerror}')

# Создание мозаик для каждого набора train/val/test
def create_mosaics(image_bbox_pairs, augmentor=None):
    processed_pairs = []
    large_images = []
    for img, bboxes in image_bbox_pairs:
        if img.size[0] < mosaic_creator.large_image_threshold and img.size[1] < mosaic_creator.large_image_threshold:
            processed_pairs.append((img, bboxes))
        else:
            large_images.append((img, bboxes))

    mosaics, large_images = mosaic_creator.create_mosaic(processed_pairs, large_images, augmentor=augmentor)
    all_images = mosaics + large_images
    return all_images

# Сохранение итоговых обучающих наборов мозаик и аннотаций
def save_mosaics(all_images, yolo_directory, bbox_directory):
    images_directory = os.path.join(yolo_directory, 'images')
    labels_directory = os.path.join(yolo_directory, 'labels')
    delete_directory(images_directory)
    os.makedirs(images_directory, exist_ok=True)
    delete_directory(labels_directory)
    os.makedirs(labels_directory, exist_ok=True)
    delete_directory(bbox_directory)
    os.makedirs(bbox_directory, exist_ok=True)
    for idx, (image, bboxes) in tqdm(enumerate(all_images), total=len(all_images), desc="Cохранение мозаик и больших картинок", unit=" unit"):
        # Сохранение мозаики
        mosaic_image_path = os.path.join(images_directory, f'image_{idx}.jpg')
        image.save(mosaic_image_path)

        # Сохранение аннотации
        annotation_path = os.path.join(labels_directory, f'image_{idx}.txt')
        with open(annotation_path, 'w') as file:
            for bbox in bboxes:
                file.write(' '.join(map(str, bbox)) + '\n')

        # Сохранение визуализации аннотированных изображений с рамками
        annotated_img = mosaic_creator.draw_source_bboxes(image, bboxes)
        annotated_img_path = os.path.join(bbox_directory, f'image_{idx}.jpg')
        annotated_img.save(annotated_img_path)

# Функция для создания yaml файла конфигурации для YOLO датасета
def create_yaml_file(dst_dir: str, class_lst):
    # Создание пути и имени файла конфигурации
    data_yaml_fname = os.path.join(dst_dir, 'data.yaml')
    os.makedirs(dst_dir, exist_ok=True)
    try:
        # Открытие файла для записи конфигурации
        with open(data_yaml_fname, 'w', encoding='utf-8') as file_yaml:
            # Запись путей к наборам данных для тренировки, валидации и тестирования
            file_yaml.write('train: ../train/images\n')
            file_yaml.write('val: ../valid/images\n')
            file_yaml.write('test: ../test/images\n\n')
            # Запись количества классов
            file_yaml.write(f'nc: {len(class_lst)}\n\n')
            # Запись списка классов
            file_yaml.write(f'names: {class_lst}')
    except Exception as e:
        print(f'[ERROR] Error writing the list to a file: {e}')

# Функция чтения списка классов
def read_classes(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as file:
        classes = [line.strip() for line in file if line.strip()]
    return classes  

# Глобальные переменные для фиксации проверочных и тестовых наборов dataloaders
global_valid_loader = None
global_test_loader = None
   
# Функция для первичного и последующего создания DataLoader'ов
def initialize_dataloaders(first_epoch, batch_size=4):
    global global_valid_loader, global_test_loader, valid_dataset, test_dataset

    # При первом вызове создаем все DataLoader'ы
    if first_epoch or not global_valid_loader or not global_test_loader:
        # Создаем мозаики для каждого набора данных
        valid_mosaics = create_mosaics(valid_set, augmentor=None)
        test_mosaics = create_mosaics(test_set, augmentor=None)

        # Инициализация YoloDataset с мозаиками
        valid_dataset = YoloDataset(valid_mosaics)
        test_dataset = YoloDataset(test_mosaics)

        # Создание DataLoader'ов для valid и test
        global_valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        global_test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Создаем train_loader в любом случае
    train_mosaics = create_mosaics(train_set, augmentor=augmentor)
    train_dataset = YoloDataset(train_mosaics)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return {
        "Обучение": train_loader, 
        "Валидация": global_valid_loader, 
        "Тестирование": global_test_loader,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset
    }