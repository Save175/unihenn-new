import torch
from torchvision import transforms

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMBER_OF_SLOTS = 16384
POLY_MODULUS_DEGREE = NUMBER_OF_SLOTS * 2
INTEGER_SCALE = 20
FRACTION_SCALE = 40
SCALE = 2**40
DEPTH = 14

COEFF_MODULUS = [INTEGER_SCALE + FRACTION_SCALE] + \
                [FRACTION_SCALE] * DEPTH + \
                [INTEGER_SCALE + FRACTION_SCALE]

TRANSFORM = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5,), (0.5,))
])
