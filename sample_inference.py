import torch
from torchvision import transforms, models
from PIL import Image

class InferenceEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.class_mapping = {
            "0": "caryota_urens",
            "1": "ceiba_speciosa",
            "2": "cycas_revoluta",
            "3": "ficus_macrophyla",
            "4": "ginkgo_biloba",
            "5": "magnolia_grandiflora",
            "6": "nolina_recurvata",
            "7": "dondrocalamus_macrocumilis",
            "8": "dracaena_draco",
            "9": "enterolobium_timbouva",
            "10": "ficus_retusa",
            "11": "platanus_occidentalis"
        }

        # Reverse mapping for index to label
        self.idx_to_label = {int(k): v for k, v in self.class_mapping.items()}
        self.num_classes = len(self.idx_to_label)

        # Load model
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image_paths):
        X = []
        print("Preprocessing images...")
        if not image_paths:
            raise ValueError("No image paths provided to preprocess.")
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            X.append(img_tensor)
        if not X:
            raise ValueError("No images were loaded. Check image paths.")
        X = torch.stack(X).to(self.device)
        print(f"Total images processed: {len(X)}")
        return X

    def postprocess(self, predictions):
        import numpy as np
        predictions = predictions.cpu().detach().numpy()
        label_indices = np.argmax(predictions, axis=1)
        labels = [self.idx_to_label.get(int(x), "unknown") for x in label_indices]
        return labels

    def run_model(self, X, batch_size=512):
        """
        Run inference on tensor X in batches.
        """
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.size(0), batch_size):
                batch = X[i:i+batch_size]
                preds = self.model(batch)
                all_preds.append(preds)
        return torch.cat(all_preds, dim=0)