import argparse
import torch

from pathlib import Path

from wp1.models.model import Model
from wp1.dataset.dataset import open_and_convert_image
from wp1.utils.inference_utils import get_weight 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        default='resnet',
        help='Name of the model to use for feature extraction. See models/model_usage for available models.'
    )
    parser.add_argument(
        '--weight_dir',
        default=Path(__file__).resolve().parents[5] / 'weight_fold'
    )
    parser.add_argument(
        '--image_path',
        default = None,
        help='Path to the image to extract features from.'
    )
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights = get_weight(args.model_name, args.weight_dir)
    model = Model(args.model_name, weights, device)

    image = open_and_convert_image(args.image_path)
    vector = model.encode(image)

    print(f"Feature vector for image {args.image_path}:")
    print(vector)
    print(f"Vector size: {vector.shape}")