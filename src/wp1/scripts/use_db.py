import argparse
import torch

from pathlib import Path

from wp1.models.model import Model
from wp1.database.db import Database
from wp1.utils.inference_utils import *

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
        '--db_name',
        default = 'db',
        help='Name of the database to create or load. It will be saved in database/db/'
    )
    parser.add_argument(
        '--indexing',
        action='store_true',
        default = False,
        help='Indicate whether to create a new database or use an existing one.'
    )
    parser.add_argument(
        '--path_to_images',
        default = None,
        help='Path to the images directory for indexing.'
    )
    parser.add_argument(
        '--training',
        action='store_true',
        default = False, 
        help='Whether to train the index or not.'
    )
    args = parser.parse_args()
    print(Path(__file__).resolve().parents[5] )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights = get_weight(args.model_name, args.weight_dir)
    model = Model(args.model_name, weights, device)

    # Create the database
    if args.indexing:
        database = Database(args.db_name, model, load=False)
        if args.path_to_images is not None:
            database.add_dataset(args.path_to_images)
    else:
        database = Database(args.db_name, model, load=True)

    if args.training:
    # Train the index
        database.train_index()
    database.save()
    database.__description__()
