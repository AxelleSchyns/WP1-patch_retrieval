import argparse
import shutil
import time
import torch
import os
import numpy as np

from models.model import Model
from tqdm import tqdm
from database.db import Database
from experiment.quantitative import full_test, test_each_class
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

project_list = ['g-afog', 'g-pas', 'none', 't-pas']

def are_equivalent(proj_im, proj_retr):
    EQUIVALENT_PROJECTS = [
        {"g-afog", "g-pas"},
    ]
    for group in EQUIVALENT_PROJECTS:
        if proj_im in group and proj_retr in group:
            return True
    return False

class GeneralDataset(Dataset):
    def __init__(self, root, dataset_name, transform=None):
        self.root = root
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        self.dataset_name = dataset_name

        self.img_list = []
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.classes = subdirs
        self.classes = sorted(self.classes)
        self.conversion = {x: i for i, x in enumerate(self.classes)}

        for i in self.classes:
            for img in os.listdir(os.path.join(root, str(i))):
                self.img_list.append(os.path.join(root, str(i), img))
    
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):  
        img_path = self.img_list[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        return image, img_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', default='resnet', type=str, help='Model architecture')
    parser.add_argument(
        '--weights', default=None, type=str, help='Path to the model weights')
    parser.add_argument(
        '--data_path', default=None, type=str, help='Path to the query image or folder of images')
    parser.add_argument(
        '--filename', default=None, type=str, help='Path to the output file')
    
    args = parser.parse_args()

    # Create the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(model_name = args.model_name, weight = args.weights, device = device)

    data = GeneralDataset(root=args.data_path, dataset_name="bp_dataset")
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False,
                                         num_workers=4, pin_memory=True)
    
    t_search = t_tot = t_model = t_transfer = 0
    cpt_skipped = 0
    accuracies = np.zeros((3,3))
    for i, (image_tensor, image_path) in tqdm(enumerate(loader), desc='Applying masks', unit='mask', ncols=80, leave=False):
        class_im = os.path.basename(os.path.dirname(image_path[0]))
        if class_im == "none":
            cpt_skipped += 1
            continue
        # move image out of data_path folder
        tmp = os.path.dirname(os.path.dirname(image_path[0]))
        shutil.move(image_path[0], tmp)
        
        # Create the database and index all but the query image
        database = Database(filename="db", model=model, load=False)
        database.add_dataset(args.data_path)
        database.save()
        
        t = time.time()
        top10_path, _, t_model_tmp, t_search_tmp, t_transfer_tmp = database.search(image_tensor, 10)
        t_tot += time.time() - t
        t_model += t_model_tmp
        t_transfer += t_transfer_tmp
        t_search += t_search_tmp

        
        proj_im = None
        for p in project_list:
            if p in image_path[0]:
                proj_im = p
                break
        # Counters for majority vote
        counts = [0, 0, 0]  # same class, same project, equivalent project

        ## Loop over top-5 results
        sim = top10_path[0][0:5]
        for j, retr in enumerate(sim):
            class_retr = os.path.basename(os.path.dirname(retr))
            for p in project_list:
                if p in retr:
                    proj_retr = p
                    break
            # Define conditions dynamically
            conditions = [
                (class_retr == class_im, 0),                # same class
                (proj_retr == proj_im, 1),                  # same project
                (are_equivalent(proj_im, proj_retr) or proj_retr == proj_im, 2)     # equivalent project
            ]

            for cond, k in conditions:
                if cond:
                    if j == 0:
                        accuracies[0, k] += 1
                    if counts[k] == 0:
                        accuracies[1, k] += 1
                    counts[k] += 1

        # Majority vote accuracy
        for k, c in enumerate(counts):
            if c > 2:
                accuracies[2, k] += 1
        

        shutil.move(os.path.join(tmp, os.path.basename(image_path[0])), image_path[0])
    
    accuracies /= len(data) - cpt_skipped
    print("Accuracies            class      project     sim")
    print(f"Top-1 accuracies: {accuracies[0]}")
    print(f"Top-5 accuracies: {accuracies[1]}")
    print(f"Maj accuracies: {accuracies[2]}")