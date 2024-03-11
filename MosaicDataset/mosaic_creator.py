from PIL import Image, ImageDraw
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures

# Класс построения набора аннотированных мозаик    
class MosaicCreator:
    def __init__(self, canvas_size=640, min_image_size=40, large_image_threshold=512, process_large_images=False):
        self.canvas_size = canvas_size
        self.min_image_size = min_image_size
        self.large_image_threshold = large_image_threshold
        self.process_large_images = process_large_images

    # Функция для отрисовки bbox на исходном изображении
    def draw_source_bboxes(self, image, bboxes):
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            class_labels, x_center, y_center, width, height = bbox
            # Преобразование нормализованных координат обратно в абсолютные значения в пикселях
            absolute_x_center = x_center * image.size[0]
            absolute_y_center = y_center * image.size[1]
            absolute_width = width * image.size[0]
            absolute_height = height * image.size[1]

            # Вычисление координат углов прямоугольника
            left = absolute_x_center - absolute_width // 2
            top = absolute_y_center - absolute_height // 2
            right = absolute_x_center + absolute_width // 2
            bottom = absolute_y_center + absolute_height // 2

            draw.rectangle([left, top, right, bottom], outline="red", width=2)
        return image
    
    # Функция для нахождения пар изображение-метка        
    def find_image_label_pairs(self, src_directory):
        images = {}
        labels = {}

        # Получаем список всех файлов в директории и поддиректориях
        all_files = []
        for root, _, files in os.walk(src_directory):
            for file in files:
                all_files.append((root, file))

        # Проходим по всем файлам, используя прогресс-бар
        for root, file in tqdm(all_files, desc="Поиск пар изображений и меток", unit=" files"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                base_name = os.path.splitext(file)[0]
                images[base_name] = os.path.join(root, file)
            elif file.lower().endswith('.txt'):
                base_name = os.path.splitext(file)[0]
                labels[base_name] = os.path.join(root, file)

        # Сопоставляем изображения и метки
        pairs = []
        for base_name, img_path in images.items():
            txt_path = labels.get(base_name)
            if txt_path:
                pairs.append((img_path, txt_path))

        return pairs
    
    # Функция загрузки всех аннотаций, которые есть на выбранной картинке
    def _load_bboxes(self, bbox_file_path):
        bboxes = []
        try:
            with open(bbox_file_path, 'r') as file:
                for line in file:
                    class_labels, x_center, y_center, width, height = map(float, line.strip().split())
                    bboxes.append([int(class_labels), x_center, y_center, width, height])
        except Exception as e:
            print(f"Ошибка при чтении файла аннотаций {bbox_file_path}: {e}")
        return bboxes              

    # Функция загрузки пар картинка-аннотации
    def _process_image(self, img_path, txt_path):
        try:
            # Загружаем картинки и их аннотации
            img = Image.open(img_path)
            original_width, original_height = img.size
            bboxes = self._load_bboxes(txt_path)

            # Проверяем, нужно ли обрабатывать большие изображения отдельно
            if self.process_large_images and (original_width >= self.large_image_threshold or original_height >= self.large_image_threshold):
                # Возвращаем изображение и его аннотации без изменений
                return img, bboxes
            else:
                # Погдонка больших изображений под размер полотна
                if original_width > self.canvas_size or original_height > self.canvas_size:
                    img.thumbnail((self.canvas_size, self.canvas_size), Image.Resampling.LANCZOS)
                return img, bboxes
        except Exception as e:
            print(f"Ошибка при открытии изображения {img_path}: {e}")
            return None, []

    # Функция загрузки и обработки пар изображений и аннотаций
    def process_image_label_pairs(self, image_label_pairs):
        if not hasattr(self, 'image_bbox_pairs') or not self.image_bbox_pairs:
            # Инициализируем списки для хранения данных, если они еще не созданы или пусты
            self.image_bbox_pairs = []
            self.large_images = []

            # Обработка пар изображений и аннотаций
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._process_image, img_path, label_path): idx for idx, (img_path, label_path) in enumerate(image_label_pairs)}
                for future in tqdm(futures, total=len(image_label_pairs), desc="Предобработка изображений", unit=" images", leave=True):
                    img, bboxes = future.result()
                    if img is not None and bboxes:
                        if self.process_large_images and (img.width >= self.large_image_threshold or img.height >= self.large_image_threshold):
                            self.large_images.append((img, bboxes)) # Сохраняем большие изображения отдельно
                        else:
                            self.image_bbox_pairs.append((img, bboxes))  # Сохраняем пару изображение-аннотация
        return self.image_bbox_pairs, self.large_images        
    
    # Функция проверки возможности размещения изображения на полотно
    def _can_place_image(self, x, y, img, occupied_areas, canvas):
        # Проверяем, что изображение не выходит за границы холста
        if x + img.width > self.canvas_size or y + img.height > self.canvas_size:
            return False

        # Проверяем перекрытие с уже размещенными изображениями
        for area in occupied_areas:
            if not (x + img.width <= area['left'] or x >= area['right'] or
                    y + img.height <= area['top'] or y >= area['bottom']):
                return False

        return True
    
    # Функция дозаполнения полотна мозаики случайными изображениями
    def _fill_extra_images(self, x, y, all_images, all_bboxes, occupied_areas, canvas, next_row_y, pbar, min_x, min_y, max_x, max_y, mosaic_bboxes):
        max_attempts_per_image = 10  # Максимальное количество попыток разместить одно изображение
        max_total_attempts = 50  # Максимальное общее количество попыток для всего полотна
        total_attempts = 0  # Общий счетчик попыток

        while all_images and total_attempts < max_total_attempts:
            attempts = 0  # Счетчик попыток размещения одного изображения

            # Пытаемся разместить случайное изображение в текущей позиции
            placed = False
            tried_indices = set()
            cnt_img = np.round(np.sqrt(len(all_images)))

            while len(tried_indices) < cnt_img and not placed and attempts < max_attempts_per_image:
                img_index = random.randint(0, len(all_images) - 1)

                if img_index in tried_indices:
                    continue  # Пропускаем уже рассмотренные изображения

                tried_indices.add(img_index)
                extra_img, extra_bboxes = all_images[img_index], all_bboxes[img_index]

                # Проверяем, помещается ли изображение в оставшееся пространство на полотне
                if self._can_place_image(x, y, extra_img, occupied_areas, canvas):
                    canvas.paste(extra_img, (x, y))
                    min_x, min_y = min(min_x, x), min(min_y, y)
                    max_x, max_y = max(max_x, x + extra_img.width), max(max_y, y + extra_img.height)
                    occupied_areas.append({'left': x, 'top': y, 'right': x + extra_img.width, 'bottom': y + extra_img.height})

                    # Корректируем координаты bbox с учетом смещения изображения
                    self.shift_bbox(extra_img, extra_bboxes, x, y, mosaic_bboxes)

                    x += extra_img.width
                    next_row_y = max(next_row_y, y + extra_img.height)
                    placed = True  # Отмечаем, что изображение было успешно размещено
                    pbar.update(1)

                    del all_images[img_index]  # Удаляем из списка, чтобы не размещать повторно
                    del all_bboxes[img_index]
                else:
                    attempts += 1  # Увеличиваем счетчик попыток
            
            total_attempts += 1  # Увеличиваем общий счетчик попыток

            # Если не удалось разместить ни одно изображение, переходим на новую строку или завершаем заполнение
            if not placed:
                if y + next_row_y >= self.canvas_size or total_attempts >= max_total_attempts:  # Если нет места для новой строки или превышено максимальное количество попыток, завершаем заполнение
                    break
                else:  # Переходим на новую строку
                    x = 0
                    y = next_row_y
                    next_row_y = y

        return min_x, min_y, max_x, max_y


    # Функция центровки мозаики на полотне
    def _center_mosaic(self, canvas, min_x, min_y, max_x, max_y):
        mosaic_width = max_x - min_x
        mosaic_height = max_y - min_y
        # Рассчитываем смещения для центровки
        offset_x = (self.canvas_size - mosaic_width) // 2
        offset_y = (self.canvas_size - mosaic_height) // 2
        final_canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))
        final_canvas.paste(canvas, (offset_x, offset_y))
        return final_canvas, offset_x, offset_y
    
    # Функция коррекции координат bbox при перемещении изображения по полотну
    def shift_bbox(self, image, bboxes, x, y, mosaic_bboxes):
        for bbox in bboxes:
            class_labels, x_center, y_center, width, height = bbox
            # Преобразование нормализованных координат bbox в абсолютные значения в пикселях на исходной картинке
            absolute_x_center = x_center * image.width
            absolute_y_center = y_center * image.height
            absolute_width = width * image.width
            absolute_height = height * image.height
            # Преобразование в абсолютные координаты полотна
            absolute_x_center += x
            absolute_y_center += y
            # Преобразование bbox в относительные координаты полотна
            corrected_x_center = absolute_x_center / self.canvas_size
            corrected_y_center = absolute_y_center / self.canvas_size
            corrected_width = absolute_width / self.canvas_size
            corrected_height = absolute_height / self.canvas_size
            # Запись аннотации полотна
            corrected_bbox = [class_labels, corrected_x_center, corrected_y_center, corrected_width, corrected_height]
            mosaic_bboxes.append(corrected_bbox)    
    
    # Основная функция сборки мозаик и аннотаций
    def create_mosaic(self, image_bbox_pairs, large_images, augmentor=None):
        mosaics = []

        # Обработка больших изображений с аугментацией
        if augmentor and self.process_large_images:
            large_images = augmentor.augment_images(large_images)
            
        # Применяем аугментацию, если она включена
        if augmentor:
            image_bbox_pairs = augmentor.augment_images(image_bbox_pairs)                    

        # Сортировка пар изображение-аннотация по размеру изображения
        image_bbox_pairs.sort(key=lambda pair: pair[0].width * pair[0].height, reverse=True)
        
        # Разделение пар на отдельные списки изображений и аннотаций
        all_images, all_bboxes = zip(*image_bbox_pairs)
        # Преобразование кортежей обратно в списки
        all_images = list(all_images)
        all_bboxes = list(all_bboxes)        
        
        with tqdm(total=len(all_images), desc="Распределение картинок по мозаикам", unit=" images", leave=True) as pbar:
            while all_images:
                canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))
                mosaic_bboxes = []
                x, y = 0, 0
                next_row_y = 0
                occupied_areas = []
                min_x, min_y, max_x, max_y = self.canvas_size, self.canvas_size, 0, 0                          

                # Размещение первого элемента на полотно
                largest_img, largest_bboxes = all_images.pop(0), all_bboxes.pop(0)
                canvas.paste(largest_img, (x, y))
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x + largest_img.width), max(max_y, y + largest_img.height)               
                occupied_areas.append({'left': x, 'top': y, 'right': x + largest_img.width, 'bottom': y + largest_img.height})
                
                # Корректируем координаты bbox с учетом смещения изображения
                self.shift_bbox(largest_img, largest_bboxes, x, y, mosaic_bboxes)

                x += largest_img.width
                next_row_y = max(next_row_y, y + largest_img.height)
                pbar.update(1)
                del largest_img
                del largest_bboxes

                # Дозаполнение полотна
                min_x, min_y, max_x, max_y = self._fill_extra_images(x, y, all_images, all_bboxes, occupied_areas, canvas, next_row_y, pbar, min_x, min_y, max_x, max_y, mosaic_bboxes)

                # Центрируем мозаику
                centered_mosaic, offset_x, offset_y = self._center_mosaic(canvas, min_x, min_y, max_x, max_y)

                # Корректируем все BBox на полотне после центровки мозаики
                adjusted_mosaic_bboxes = []
                for bbox in mosaic_bboxes:
                    class_labels, x_center, y_center, width, height = bbox
                    corrected_x_center = x_center + offset_x / self.canvas_size
                    corrected_y_center = y_center + offset_y / self.canvas_size
                    corrected_bbox = [class_labels, corrected_x_center, corrected_y_center, width, height]
                    adjusted_mosaic_bboxes.append(corrected_bbox)

                mosaics.append((centered_mosaic, adjusted_mosaic_bboxes))

        return mosaics, large_images