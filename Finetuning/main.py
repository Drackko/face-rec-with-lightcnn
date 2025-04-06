import torch
import torch.serialization
from numpy._core.multiarray import _reconstruct

# Option 2: Whitelist specific functions (more secure)
torch.serialization.add_safe_globals([_reconstruct])

# Use the correct file name (.pth extension) and disable weights_only
gallery = torch.load('gallery.pth', weights_only=False)

# Print information
print(f"Number of identities: {len(gallery)}")
print(f"Identities: {list(gallery.keys())}")
print(f"Feature vector shape: {gallery[list(gallery.keys())[0]].shape}")