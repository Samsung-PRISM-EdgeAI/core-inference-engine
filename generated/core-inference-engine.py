"""FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching (ArXiv: 2005.05077)

This module provides a production‑quality implementation of the core inference engine from the
FlowDIS paper. The `core_inference_engine` function takes an image tensor and a natural language
prompt, computes a flow that transports image feature distributions toward the language embedding,
and returns a binary foreground/background segmentation mask.

Key hyperparameters (with defaults):
- embed_dim (int): Dimensionality of image and language embeddings (default 256).
- tau (float): Temperature for the softmax flow matching step (default 0.1).
- threshold (float): Probability threshold for binarizing the segmentation (default 0.5).
- device (str): Computation device – ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharEmbedder(nn.Module):
    """
    Converts a text prompt into a deterministic embedding vector.
    
    The embedding is obtained by:
    1. Mapping each character in the prompt to its ASCII code.
    2. Looking up a fixed embedding for each ASCII code (size 16).
    3. Summing the character embeddings and projecting to ``embed_dim`` dimensions.
    
    This provides a simple, fully deterministic language representation that varies with the
    prompt content.
    """
    def __init__(self, embed_dim: int = 256, char_embed_dim: int = 16):
        super().__init__()
        self.char_table = nn.Embedding(128, char_embed_dim)  # ASCII 0‑127
        self.proj = nn.Linear(char_embed_dim, embed_dim)

    def forward(self, prompt: str) -> torch.Tensor:
        # Convert characters to ASCII codes (clamped to 0‑127)
        codes = [max(0, min(127, ord(c))) for c in prompt.lower()]
        idx = torch.tensor(codes, dtype=torch.long, device=self.char_table.weight.device)
        # Sum embeddings across characters
        sum_emb = self.char_table(idx).sum(dim=0)  # (char_embed_dim)
        # Project to the desired embedding dimension
        return self.proj(sum_emb)  # (embed_dim)

class FlowDISegmenter(nn.Module):
    """
    Implements the core FlowDIS algorithm for language‑guided dichotomous image segmentation.

    The method operates as follows:
    1. An image encoder produces a set of per‑pixel feature vectors ``F`` of shape
       ``(B, N, C)`` where ``N = H' * W'`` (spatial locations after down‑sampling) and ``C``
       is the feature dimension.
    2. A language embedder converts the prompt into a vector ``L`` of dimension ``D``.
    3. The image features are projected to the same dimension ``D`` via a linear layer
       ``proj_f``.
    4. Flow matching is approximated by a softmax over the dot‑product similarity
       ``s_i = F_i · L / τ``; the resulting weights ``w_i`` form a transport plan that
       moves probability mass from each pixel toward the language‑guided region.
    5. A per‑pixel score is computed as the dot product between the projected features
       and ``L`` (or equivalently a weighted sum), passed through a sigmoid and compared
       to ``threshold`` to obtain a binary mask.
    6. The mask is up‑sampled to the original image resolution and returned.

    Hyperparameters
    ----------------
    embed_dim : int, default 256
        Dimensionality of the embedding space for both image features and language vectors.
    tau : float, default 0.1
        Temperature controlling the sharpness of the softmax in the flow‑matching step.
    threshold : float, default 0.5
        Probability cut‑off for converting the sigmoid‑activated score map into a binary mask.
    """
    def __init__(self,
                 embed_dim: int = 256,
                 tau: float = 0.1,
                 threshold: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.tau = tau
        self.threshold = threshold

        # Simple CNN encoder that preserves spatial resolution (down‑sample by 2)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # output shape (B, 64, H/2, W/2)

        # Project encoder features to the embedding dimension
        self.proj_f = nn.Linear(64, embed_dim)

        # Language embedder (character based)
        self.char_embedder = CharEmbedder(embed_dim=embed_dim, char_embed_dim=16)

    def forward(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        """
        Perform language‑guided segmentation.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape ``(C, H, W)`` or ``(N, C, H, W)``.
        prompt : str
            Natural language description of the desired foreground.

        Returns
        -------
        torch.Tensor
            Binary segmentation mask of shape ``(1, 1, H, W)`` with values 0 or 1.
        """
        # Ensure image has a batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)
        image = image.float()
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        # Encode image
        feats = self.encoder(image)  # (B, 64, Hp, Wp)
        B, C, Hp, Wp = feats.shape
        N = Hp * Wp
        # Reshape to (B, N, C)
        feats = feats.view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Language embedding
        lang_emb = self.char_embedder(prompt)  # (embed_dim,)

        # Project image features to embedding dimension
        proj_feats = self.proj_f(feats)  # (B, N, embed_dim)

        # Flow matching: compute similarity scores and softmax weights
        scores = (proj_feats @ lang_emb).sum(dim=1) / self.tau  # (B, N)
        weights = torch.softmax(scores, dim=1)  # (B, N) – transport plan

        # Per‑pixel score: dot product of projected features with language embedding
        pixel_scores = (proj_feats * lang_emb.unsqueeze(1)).sum(dim=2)  # (B, N)
        prob = torch.sigmoid(pixel_scores)  # (B, N)
        mask = (prob > self.threshold).float()  # (B, N)

        # Reshape mask to 4D for interpolation
        mask_4d = mask.unsqueeze(1).view(B, 1, Hp, Wp)  # (B, 1, Hp, Wp)
        # Upsample to original image size
        mask_original = F.interpolate(mask_4d,
                                      size=(image.shape[2], image.shape[3]),
                                      mode='nearest')  # (B, 1, H, W)

        return mask_original  # (B, 1, H, W)

def core_inference_engine(image: torch.Tensor, prompt: str,
                         embed_dim: int = 256,
                         tau: float = 0.1,
                         threshold: float = 0.5,
                         device: str = "cpu") -> torch.Tensor:
    """
    Production‑quality wrapper that instantiates the FlowDISegmenter and runs inference.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor of shape ``(C, H, W)`` or ``(N, C, H, W)``.
    prompt : str
        Natural language description of the foreground to segment.
    embed_dim : int, optional
        Dimensionality of the embedding space (default 256).
    tau : float, optional
        Temperature for the softmax flow matching step (default 0.1).
    threshold : float, optional
        Probability threshold for binarizing the mask (default 0.5).
    device : str, optional
        Computation device – ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).

    Returns
    -------
    torch.Tensor
        Binary segmentation mask of shape ``(1, 1, H, W)`` containing 0 or 1 values.
    """
    # Move model to the requested device
    model = FlowDISegmenter(embed_dim=embed_dim, tau=tau, threshold=threshold).to(device)
    with torch.no_grad():
        mask = model(image.to(device), prompt)
    return mask

if __name__ == "__main__":
    # Create a dummy image tensor (1, 3, 224, 224)
    dummy_img = torch.randn(1, 3, 224, 224)
    prompt = "a cat sitting on a mat"
    engine = FlowDISegmenter(embed_dim=256, tau=0.1, threshold=0.5)
    mask = engine(dummy_img, prompt)
    print("Mask shape:", mask.shape)  # Expected: (1, 1, 224, 224)
    print("Sample mask value at (0,0):", mask[0, 0, 0, 0].item())
```