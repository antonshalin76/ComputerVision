# Dynamic YOLO Mosaic Generator

## Описание
Dynamic YOLO Mosaic Generator – это мощный инструмент для создания динамических мозаик изображений с аннотациями, предназначенный для улучшения процесса обучения моделей компьютерного зрения, таких как YOLO. Этот пакет включает в себя улучшенные функции аугментации, стратифицированный сплиттер датасета и эффективное формирование мозаик.

### Основные функции
- **Эффективное создание мозаик**: На каждое полотно мозаики первым выкладывается наибольшее по площади изображение. Полотно дозаполняется для максимальной плотности расположения.
- **Многопоточная обработка**: Ускоренное формирование мозаик благодаря многопоточности на этапе масштабирования картинок.
- **Динамическая аугментация**: Разнообразные аугментации из библиотеки albumentations, активируемые параметром-переключателем.
- **Улучшенная обработка больших изображений**: Возможность обработки крупных изображений без изменения их размера и наложения на полотно.
- **Режим исключения маленьких изображений**: Возможность исключить из включения в датасет слишком маленьких изображений.
- **Стратифицированный сплиттер**: Равномерное распределение классов сущностей между наборами train/valid/test.
- **Генерация конфигурационных файлов**: Создание yaml-файлов для управления датасетами.
- **PyTorch Dataset интеграция**: Класс Dataset для использования в обучающих циклах PyTorch.

## Установка

```
git clone https://github.com/antonshalin76/ComputerVision.git
```
```
cd ComputerVision
```
```
pip install .
```

## Примеры использования

Пример создания мозаик с аннотациями:

```python
from dynamic_yolo_mosaic_generator import MosaicCreator, ImageAugmentor, SplitSubset, delete_directory, read_classes, create_yaml_file, initialize_dataloaders, save_mosaics
import os

# Рабочие пути
src_directory = "path/to/source" # Путь с исходными данными. По этому же пути должен лежать файл списка классов classes.txt
dst_directory = "path/to/destination" # Путь к папке, где будут сохраняться разделенные наборы данных
yolo_directory = "path/to/yolo" # Путь для yolo мозаик
bbox_directory = "path/to/bbox" # Путь для визуализации мозаик с наложенными bbox

# Формирование рабочих директорий
for directory in [dst_directory, yolo_directory, bbox_directory]:
    delete_directory(directory)
    os.makedirs(directory, exist_ok=True)

# Создание экземпляров классов
augmentor = ImageAugmentor()
mosaic_creator = MosaicCreator(canvas_size=640, min_image_size=40, large_image_threshold=1280, process_large_images=False)

# Чтение списка классов из файла
try:
    class_lst = read_classes(os.path.join(src_directory, 'classes.txt'))
except Exception as e:
    print(f"Ошибка при чтении файла classes.txt: {e}")
    class_lst = []

if class_lst:
    # Создание конфигурационных файлов
    create_yaml_file(dst_directory, class_lst)
    create_yaml_file(yolo_directory, class_lst)

    # Загрузка пар изображение-метка
    image_label_path = mosaic_creator.find_image_label_pairs(src_directory)
    image_bbox_pairs, large_images = mosaic_creator.process_image_label_pairs(image_label_path)
    all_image_bbox_pairs = image_bbox_pairs + large_images

    # Распределение исходных аннотированных пар на три набора обучения
    splitter = SplitSubset(all_image_bbox_pairs, split_ratio=(0.7, 0.2, 0.1)) # Задаем пропорции наборов данных

    # Проверка наличия существующих наборов данных в памяти
    data_folders = {'train': os.path.join(dst_directory, 'train'), 
                    'valid': os.path.join(dst_directory, 'valid'), 
                    'test': os.path.join(dst_directory, 'test')}

    train_set, valid_set, test_set = None, None, None

    # Проверяем, существуют ли сохраненные папки с наборами данных
    if all(os.path.exists(folder) for folder in data_folders.values()):
        train_set, valid_set, test_set = splitter.load_sets_from_folders(data_folders)
    else:
        # Создаем и сохраняем новые наборы данных из исходного набора
        splitter.save_splits(dst_directory)
        train_set, valid_set, test_set = splitter.load_sets_from_folders(data_folders)

# Параметры обучения

batch_size = 32
epochs = 10

# Создание DataLoader'ов для первой эпохи
dataloaders = initialize_dataloaders(first_epoch=True, batch_size=batch_size)

# Создание DataLoader'ов для последующих эпох (эмуляция обучения)
for i in range(epochs-1):
    dataloaders = initialize_dataloaders(first_epoch=False, batch_size=batch_size)
print(' Batches для обучения: ', len(dataloaders["Обучение"]),'\n',
      'Batches для валидации: ', len(dataloaders["Валидация"]),'\n',
      'Batches для тестирования: ', len(dataloaders["Тестирование"]),'\n', '\n',
      'Мозаик для обучения: ', len(dataloaders["train_dataset"]),'\n',
      'Мозаик для валидации: ', len(dataloaders["valid_dataset"]),'\n',
      'Мозаик для тестирования: ', len(dataloaders["test_dataset"]))

# ==================================== Контроль ===================================== #

# Проверка количества изображений в каждом наборе
print(f"Train set: {len(train_set)} images")
print(f"Validation set: {len(valid_set)} images")
print(f"Test set: {len(test_set)} images\n")

# Проверим стратификацию поднаборов по классам
train_annotations = splitter.count_annotations_by_class(train_set)
valid_annotations = splitter.count_annotations_by_class(valid_set)
test_annotations = splitter.count_annotations_by_class(test_set)
print("Количество аннотаций в train наборе:", train_annotations)
print("Количество аннотаций в valid наборе:", valid_annotations)
print("Количество аннотаций в test наборе:", test_annotations)

# Сохранение наборов аннотированных мозаик по структуре папок для визуального контроля. Для обучения убрать.
save_mosaics(dataloaders["train_dataset"], os.path.join(yolo_directory, 'train'), os.path.join(bbox_directory, 'train'))
save_mosaics(dataloaders["valid_dataset"], os.path.join(yolo_directory, 'valid'), os.path.join(bbox_directory, 'valid'))
save_mosaics(dataloaders["test_dataset"], os.path.join(yolo_directory, 'test'), os.path.join(bbox_directory, 'test'))
```
## Контакты
Для вопросов или предложений: anton.shalin@gmail.com.
<a href="https://github.com/antonshalin76">GitHub профиль</a>
