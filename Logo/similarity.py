import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from typing import List, Tuple

class ImageMatcher:
    def __init__(self):
        """
        Initializes the matcher by loading the ResNet-152 model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing ImageMatcher with resnet152 model on device: {self.device}")
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
        self.transform_rgb = T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            T.ToTensor()
        ])

    def find_best_matches_from_memory(self, 
                                      ref_images: List[Image.Image], 
                                      ref_blank_masks: List[Image.Image], 
                                      candidate_images: List[Image.Image], 
                                      candidate_masks: List[Image.Image], 
                                      candidate_blank_masks: List[Image.Image],
                                      candidate_bboxes: List[List[int]], 
                                      threshold: float) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image],List[Image.Image], List[Image.Image], List]:
        """
        Finds the best match for each reference image from the candidates and filters
        out any matches below the threshold.
        """
        if not ref_images or not candidate_images:
            print("Error: One or both of the image lists are empty.")
            return [], [], [], [], []

        # Prepare batches of tensors
        ref_batch = torch.stack([self.transform_rgb(img.convert('RGB')) for img in ref_images]).to(self.device)
        candidate_batch = torch.stack([self.transform_rgb(img.convert('RGB')) for img in candidate_images]).to(self.device)
        with torch.no_grad():
            embeddings_ref = self.model(ref_batch).flatten(start_dim=1)
            embeddings_cand = self.model(candidate_batch).flatten(start_dim=1)
            similarity_matrix = torch.nn.functional.cosine_similarity(
                embeddings_ref.unsqueeze(1),
                embeddings_cand.unsqueeze(0),
                dim=2
            )
        
        # Find the best candidate match for each reference image
        best_scores, best_indices = torch.max(similarity_matrix, dim=1)
        
        # Filter matches based on the threshold
        match_mask = best_scores >= threshold
        matched_ref_indices = torch.where(match_mask)[0]
        
        if matched_ref_indices.shape[0] == 0:
            print("Warning: No matches found above the threshold.")
            return [], [], [], [], []
        
        matched_cand_indices = best_indices[match_mask]
        
        final_ref_images = [ref_images[i] for i in matched_ref_indices.cpu().tolist()]
        final_ref_blank_masks = [ref_blank_masks[i] for i in matched_ref_indices.cpu().tolist()]
        final_images = [candidate_images[i] for i in matched_cand_indices.cpu().tolist()]
        final_masks = [candidate_masks[i] for i in matched_cand_indices.cpu().tolist()]
        final_blank_masks = [candidate_blank_masks[i] for i in matched_cand_indices.cpu().tolist()]
        final_bboxes = [candidate_bboxes[i] for i in matched_cand_indices.cpu().tolist()]
        
        return final_ref_images,final_ref_blank_masks, final_images, final_masks, final_blank_masks, final_bboxes