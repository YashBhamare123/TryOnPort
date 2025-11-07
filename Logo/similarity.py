import torch
import torchvision.models as models
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, Normalize, ToDtype
from typing import List, Tuple

class ImageMatcher:
    """
    A class for finding the best matching images between two sets using deep learning embeddings.
    It uses a pre-trained ResNet-152 model to generate feature vectors for images and
    computes cosine similarity to find matches.
    """

    def __init__(self):
        """
        Initializes the matcher by loading the ResNet-152 model and defining the image transformation pipeline.
        The model is set to evaluation mode, moved to the available device, and cast to bfloat16 for performance.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16
        print(f"Initializing ImageMatcher with ResNet-152 model on device: {self.device}")

        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
        # Remove the final classification layer to get feature embeddings
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device, dtype=self.dtype).eval()

        # Define the transformation pipeline for input images
        self.transform = Compose([
            ToDtype(torch.float32, scale=True), # Normalize expects float32
            Resize(256, antialias=True),
            CenterCrop(224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToDtype(self.dtype, scale=False) # Convert back to bfloat16
        ])

    def find_best_matches(self,
                          ref_images: torch.Tensor,
                          ref_blank_masks: torch.Tensor,
                          candidate_images: torch.Tensor,
                          candidate_masks: torch.Tensor,
                          candidate_blank_masks: torch.Tensor,
                          candidate_bboxes: List[List[int]],
                          threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Finds the best match for each reference image from a set of candidate images.

        It computes embeddings for all reference and candidate images, calculates the cosine
        similarity matrix between them, and for each reference image, finds the candidate with
        the highest similarity. Matches with a similarity score below the threshold are discarded.

        Args:
            ref_images (torch.Tensor): A batch of reference images, shape (B_ref, C, H, W).
            ref_blank_masks (torch.Tensor): A batch of blank masks for reference images.
            candidate_images (torch.Tensor): A batch of candidate images, shape (B_cand, C, H, W).
            candidate_masks (torch.Tensor): A batch of masks for candidate images.
            candidate_blank_masks (torch.Tensor): A batch of blank masks for candidate images.
            candidate_bboxes (List[List[int]]): Bounding boxes for the candidate images.
            threshold (float): The minimum cosine similarity score to consider a match.

        Returns:
            A tuple containing the filtered tensors and bounding boxes for the successful matches:
            - (torch.Tensor): Matched reference images.
            - (torch.Tensor): Matched reference blank masks.
            - (torch.Tensor): Matched candidate images.
            - (torch.Tensor): Matched candidate masks.
            - (torch.Tensor): Matched candidate blank masks.
            - (List): Matched candidate bounding boxes.
        """
        if ref_images.shape[0] == 0 or candidate_images.shape[0] == 0:
            print("Warning: One or both of the input tensor batches are empty.")
            empty_tensor = torch.empty(0, *ref_images.shape[1:], device=self.device, dtype=self.dtype)
            empty_mask = torch.empty(0, *ref_blank_masks.shape[1:], device=self.device, dtype=self.dtype)
            return empty_tensor, empty_mask, empty_tensor, empty_mask, empty_mask, []

        # Apply transformations and compute embeddings
        ref_batch = self.transform(ref_images.to(self.device))
        candidate_batch = self.transform(candidate_images.to(self.device))

        with torch.no_grad():
            embeddings_ref = self.model(ref_batch).flatten(start_dim=1)
            embeddings_cand = self.model(candidate_batch).flatten(start_dim=1)
            # Calculate cosine similarity between each reference and all candidates
            similarity_matrix = torch.nn.functional.cosine_similarity(
                embeddings_ref.unsqueeze(1),
                embeddings_cand.unsqueeze(0),
                dim=2
            )

        # Find the best candidate match for each reference image
        best_scores, best_indices = torch.max(similarity_matrix, dim=1)

        # Filter matches based on the similarity threshold
        match_mask = best_scores >= threshold
        matched_ref_indices = torch.where(match_mask)[0]

        if matched_ref_indices.shape[0] == 0:
            print("Warning: No matches found above the threshold.")
            empty_tensor = torch.empty(0, *ref_images.shape[1:], device=self.device, dtype=self.dtype)
            empty_mask = torch.empty(0, *ref_blank_masks.shape[1:], device=self.device, dtype=self.dtype)
            return empty_tensor, empty_mask, empty_tensor, empty_mask, empty_mask, []

        matched_cand_indices = best_indices[match_mask]

        # Gather the final matched items
        final_ref_images = ref_images[matched_ref_indices]
        final_ref_blank_masks = ref_blank_masks[matched_ref_indices]
        final_images = candidate_images[matched_cand_indices]
        final_masks = candidate_masks[matched_cand_indices]
        final_blank_masks = candidate_blank_masks[matched_cand_indices]
        final_bboxes = [candidate_bboxes[i] for i in matched_cand_indices.cpu().tolist()]

        return final_ref_images, final_ref_blank_masks, final_images, final_masks, final_blank_masks, final_bboxes
