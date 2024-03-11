from torch.utils.data import Dataset

# Класс Dataset для работы с DataLoader        
class YoloDataset(Dataset):
    def __init__(self, image_bbox_pairs):
        self.image_bbox_pairs = image_bbox_pairs

    def __len__(self):
        return len(self.image_bbox_pairs)

    def __getitem__(self, idx):
        image, bboxes = self.image_bbox_pairs[idx]
        return image, bboxes