from PIL import Image
import matplotlib.pyplot as plt
import torch

def show_tensor(ts : torch.Tensor, id : str = 'Figure') -> None:
    ts = ts[0]
    plt.imshow(ts.permute(1, 2, 0))
    plt.title(id)
    plt.show()