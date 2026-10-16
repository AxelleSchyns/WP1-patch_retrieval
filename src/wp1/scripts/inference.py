import argparse
import torch

from pathlib import Path

from wp1.models.model import Model
from wp1.database.db import Database
from wp1.experiment.quantitative import full_test, test_each_class
from wp1.utils.inference_utils import get_weight

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', default='resnet', type=str, help='Name of the model to use for feature extraction. See models/model_usage for available models.')
    parser.add_argument(
        '--weight_dir', default=Path(__file__).resolve().parents[5] / 'weight_fold', type=str, help='Path to the directory containing the model weights')
    parser.add_argument(
        '--db_name', default='db', type=str, help='Name of the database')
    parser.add_argument(
        '--indexing', action='store_true', help='If set, the database will be indexed with the dataset at path_index')
    parser.add_argument(
        '--path_index', default=None, type=str, help='Path to the dataset to index')
    parser.add_argument(
        '--query_path', default=None, type=str, help='Path to folder of query images')
    parser.add_argument(
        '--measure', default='all', type=str, help='Measure to use among: all, remove (uliege only), weighted, random')
    parser.add_argument(
        '--filename', default="result", type=str, help='Path to the output file')
    parser.add_argument(
        '--stat', action='store_true', help='If set, statistics will be computed on the results')
    parser.add_argument(
        '--class_split', action='store_true', help='If set, the results will be computed for each class separately (uliege only)')
    parser.add_argument(
        '--project_name', default=None, type=str, help='Name of the project to use for inference')
    parser.add_argument(
        '--class_name', default=None, type=str, help='Name of the class to use for inference')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create the model
    weight = get_weight(args.model_name, args.weight_dir)
    model = Model(model_name = args.model_name, weight = weight, device = device)

    # Create the database
    database = Database(filename=args.db_name, model=model, load=not args.indexing, device=device)
    if args.indexing:
        if args.path_index is None:
            print("Please provide a path to the dataset to index with --path_index")
            exit(-1)
        database.add_dataset(args.path_index)
        database.save()
    
    # check validity of the query dataset and the indexed dataset
    assert args.query_path is not None, "Please provide a path to the query image or folder of images with --query_path"
    data_name_query = Path(args.query_path).parent.name 
    data_name_index = Path(args.path_index).parent.name if args.path_index is not None else Path(database.data_path[0]).parent.name
    if data_name_query != data_name_index:
        print(f"Warning: The query dataset ({data_name_query}) is different from the indexed dataset ({data_name_index}). This may lead to unexpected results.")
        exit(-1)
    else:
        data_name = data_name_query
    
    # Perform inference
    if args.class_split:
        test_each_class(model, args.db_name, data_name, args.query_path, args.measure, "./logs/results_per_class.xlsx")
    else:
        full_test(model, args.db_name, data_name, args.query_path, args.measure, log_filename=args.filename, stat = args.stat, project_name=args.project_name, class_name=args.class_name)

        """display_cm(model.weight, args.measure, ground_truth, data, predictions)
        display_precision_recall(model.weight, args.measure, ground_truth, predictions[0])
        display_prec_im(model.weight, props, data, args.measure)"""