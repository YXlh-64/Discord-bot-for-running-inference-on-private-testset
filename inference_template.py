# make sure to import the necessary libraries for your model

class InferenceEngine:
    def __init__(self, user_id, model_path):
        # RO section
        self.user_id = user_id
        self.model_path = model_path
        # End of RO section

    def preprocess(self, image_paths):
        """
        Convert list of image paths to PyTorch tensors
            Args: image_paths (list of str): List of file paths to images
            Returns: torch.Tensor: Batch of images as a tensor
        """
        pass

    def postprocess(self, predictions):
        """
        Convert model outputs to final labels
            Args: predictions (torch.Tensor): Model output tensor
            Returns: pd.Series: Series of predicted labels
        """
        pass


    def run_model(self):
        """
        Run inference on preprocessed images
            Returns: torch.Tensor: Model predictions
        """
        pass