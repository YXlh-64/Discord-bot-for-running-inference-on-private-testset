import os
from tqdm import tqdm
from inference import InferenceEngine
from utils import prepare_test_set
import torch
import pandas as pd
from utils import prepare_image_ids, prepare_test_set
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def save_predictions(labels, image_paths):
    print("save_predictions called.")

    filenames = prepare_image_ids(image_paths=image_paths)

    predictions = pd.DataFrame({
        "image": filenames,
        "label": labels
    })
    print("DataFrame created:")

    submission_path = "predictions.csv"
    print("Current working directory:", os.getcwd())
    print("Attempting to save to:", submission_path)
    try:
        predictions.to_csv(submission_path, index=False)
    except Exception as e:
        print(f"Error saving predictions: {e}")

    return submission_path

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, img_path

def main():
    try:
        print(f"The script is running inside a Docker container {os.uname().nodename}")
        model_path = "model.pth"

        print("Preparing test set...")
        image_paths = prepare_test_set()

        print("Initializing inference engine...")
        engine = InferenceEngine(model_path)
        BATCH_SIZE = 64

        print("Creating dataset and dataloader...")
        dataset = ImageDataset(image_paths, engine.transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        all_labels = []
        all_image_paths = []

        print("Running inference with DataLoader...")
        for batch_imgs, batch_paths in tqdm(dataloader, total=len(dataloader), desc="Inference", leave=True):
            batch_imgs = batch_imgs.to(engine.device)
            preds = engine.run_model(batch_imgs)
            labels = engine.postprocess(preds)
            all_labels.extend(labels)
            all_image_paths.extend(batch_paths)
            torch.cuda.empty_cache()

        print("Saving predictions...")
        submission_path = save_predictions(all_labels, all_image_paths)
        print(f"Predictions saved to {submission_path}")

    except Exception as e:
        import traceback
        error_path = "error.txt"
        with open(error_path, "w") as f:
            f.write("An error occurred during inference:\n")
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print(f"Error written to {error_path}")

if __name__ == "__main__":
    main()