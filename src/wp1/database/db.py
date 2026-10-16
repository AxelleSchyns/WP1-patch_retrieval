
import faiss
import torch
import redis
import time
import gc
import json
import sys

import numpy as np

from tqdm import tqdm
from wp1.dataset.dataset import AddDataset
from wp1.models.model import Model
from wp1.utils.inference_utils import get_weight
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DB_DIRECTORY = BASE_DIR / "db"

# Class that represents the database
class Database:
    def __init__(self, filename, model=None, load=False, device='cuda:0'):
        self.filename = str(DB_DIRECTORY / filename)
        self.data_path = []
        self.device = device
        self.trained = False

        res = faiss.StandardGpuResources()  # Allocation of streams and temporary memory 

        # Load an existing database
        if load:
            try:
                self.index = faiss.read_index(self.filename)
                self.r = redis.Redis(host='localhost', port=6379, db=0)

                json_path = f"{self.filename}.json"
                with open(json_path, 'r') as f:
                    payload = json.load(f)

                self.vector_size = payload['vector_size']
                self.data_path = payload['data_path']
                self.device = payload['device']
                self.trained = payload['trained']

                self.model = Model(payload['model'], payload['weight_path'], self.device) if payload['model'] else None
                if model and self.model.model_name != model.model_name:
                    print(f"Warning: The model used to create the database ({self.model.model_name}) is different from the provided model ({model.model_name}).")
                    print("The database will be used with the model it was created with.")
                elif self.model is None:
                    print("Warning: No model information found in the database metadata and no model provided. Exiting")
                    sys.exit(1)
                else: # just in case - but should not be triggered as model is saved in all db_files
                    self.model = model

                redis_data = payload.get('redis_data', {})
                if redis_data:
                        self.r.mset(redis_data)
                print(f"Loaded {len(redis_data)} keys into Redis.")
            except Exception as e:
                print(self.filename, json_path)
                print(f"Failed to load database: {e}")
                sys.exit(1)
        else:
            # Create a new database
            print("Creating a new FAISS index")
            if model is None:
                print("Error: No model provided for creating a new database. Exiting.")
                sys.exit(1)
            self.model = model
            self.vector_size = model.num_features 
            
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.index = faiss.IndexIDMap(self.index)
            self.r = redis.Redis(host='localhost', port=6379, db=0)
            self.r.flushdb()
            self.r.set('last_id', 0)  # Set a value in Redis with key = last_id
        if 'cuda' in device:
            print("Transferring index to GPU")
            try:
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("Index transferred to GPU")
            except Exception as e:
                print(f"Failed to transfer index to GPU: {e}")
    def __len__(self):
        return self.index.ntotal
    
    def __description__(self):
        indexing = "trained" if self.trained else "untrained / brute_force"
        return f"Database with {self.index.ntotal} vectors of size {self.vector_size} on device {self.device}. \n \
            Vectors are obtained from model {self.model.model_name} using images from {self.data_path} and stored in FAISS index {self.filename}. \n \
            Indexing method: {indexing}"

    # x = vector of images 
    def add(self, x, names):
        last_id = int(self.r.get('last_id').decode('utf-8'))
        self.index.add_with_ids(x, np.arange(last_id, last_id + x.shape[0])) 
        # Previous command not working on cpu ! work when adding one by one
        for n, x_  in zip(names, x):
            if n is None or n == '':
                print(f"Warning: Skipping image with None or empty name.")
                continue  # Skip if the name is None or empty
            self.r.set(str(last_id), n) # Set the name of the image at key = id
            self.r.set(n, str(last_id)) # Set the id of the image at key = name 
            last_id += 1

        self.r.set('last_id', last_id) # Update the last id to take into account the added images

    @torch.no_grad()
    def add_dataset(self, data_root):
        model = self.model
        model_name = self.model.model_name
        model_device = next(model.parameters()).device

        if data_root not in self.data_path:
            self.data_path.append(data_root)
        data = AddDataset(data_root, model_name)
        loader = torch.utils.data.DataLoader(
            data, batch_size=32, num_workers=8, pin_memory=True, timeout=30, persistent_workers=False )

        
        t_model, t_indexing, t_transfer = 0, 0, 0

        def log_mem(step=""):
            torch.cuda.empty_cache()
            print(f"[{step}] CUDA mem alloc: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
            print(f"[{step}] CUDA mem reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        for i, (images, filenames) in tqdm(enumerate(loader), total=len(loader)):
            if i % 500 == 0:
                log_mem()

            # Ensure shape
            images = images.view(-1, 3, 224, 224).to(device=model_device, non_blocking=True)

            try:
                t0 = time.time()

                out = model.encode(images)

                out = out.contiguous()

                torch.cuda.synchronize()
                t_model += time.time() - t0

                # Check for NaNs
                if torch.isnan(out).any():
                    print(f"NaNs in batch {i}")
                    continue

                # Transfer to CPU
                t = time.time()
                out_np = out.cpu().numpy()
                torch.cuda.synchronize()
                t_transfer += time.time() - t

                # Indexing / saving
                t = time.time()
                self.add(out_np, list(filenames))
                torch.cuda.synchronize()
                t_indexing += time.time() - t

            except RuntimeError as e:
                print(f" Runtime error at batch {i}: {e}")
                log_mem(f"Error @ batch {i}")
                continue
            # Optional memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        print(f"Time - model: {t_model:.2f}s | transfer: {t_transfer:.2f}s | indexing: {t_indexing:.2f}s")
        self.save()

    @torch.no_grad()
    def search(self, query_im, nrt_neigh=10):
        model = self.model
        model_device = next(model.parameters()).device

        image = query_im.to(device=model_device, non_blocking=True).reshape(-1, 3, 224, 224)

        t0_model = time.time()

        out = model.encode(image)

        torch.cuda.synchronize()
        t_model = time.time() - t0_model

        # Transfer to CPU
        torch.cuda.synchronize()
        t0_transfer = time.time()
        out = out.cpu().numpy()
        t_transfer = time.time() - t0_transfer

        # Search
        torch.cuda.synchronize()
        t0_search = time.time()
        distance, labels = self.index.search(out, nrt_neigh)
        labels = [l for l in list(labels[0]) if l != -1]
        values = [self.r.get(str(l)).decode("utf-8") for l in labels]
        t_search = time.time() - t0_search

        return values, distance[0], t_model, t_search, t_transfer

    def remove(self, name):
        key = self.r.get(name).decode('utf-8')
        label = int(key)

        idsel = faiss.IDSelectorRange(label, label+1)

        if 'cuda' in self.device:
            self.index = faiss.index_gpu_to_cpu(self.index)
        self.index.remove_ids(idsel)
        self.save()
        self.r.delete(key)
        self.r.delete(name)
        if 'cuda' in self.device:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # TODO: save the free indexes and reuse them instead of incrementing the last_id

    def train_index(self):
        if self.trained:
            print("Index is already trained. Skipping training.")
            return
        if 'cuda' in self.device:
            self.index = faiss.index_gpu_to_cpu(self.index)
        batch_size = 128
        x = []
        keys = []
        all_keys = self.r.keys("*")
        for k in all_keys:
            k = k.decode("utf-8")
            # Only keep the indexes as keys, not the names nor last_id 
            if k.isdigit():
                index = int(k)
                keys.append(index)
                vec = self.index.index.reconstruct(index)
                x.append(vec)
        if len(x) >= 10:
            num_clusters = int(np.sqrt(self.index.ntotal))

            self.quantizer = faiss.IndexFlatL2(self.vector_size)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.vector_size,
                                                    num_clusters)

            if 'cuda' in self.device:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

            x = np.asarray(x, dtype=np.float32)
            self.index.train(x)
            self.index.nprobe = num_clusters // 10

            num_batches = self.index.ntotal // batch_size

            for i in range(num_batches+1):
                if i == num_batches:
                    x_ = x[i * batch_size:, :]
                    key = keys[i * batch_size:]
                else:
                    x_ = x[i * batch_size: (i + 1) * batch_size, :]
                    key = keys[i * batch_size: (i+1) * batch_size]
                self.index.add_with_ids(x_, np.array(key, dtype=np.int64))
        self.trained = True

    def save(self):
        # 1. Save FAISS index
        if 'cuda' in self.device: 
            faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.filename)
        else:
            faiss.write_index(self.index, self.filename)
            
        # 2. Save Redis mappings to a JSON file alongside the index
        redis_data = {k.decode('utf-8'): self.r.get(k).decode('utf-8') for k in self.r.keys('*')}
        payload = {
            'index': self.filename,
            'data_path': self.data_path,
            'vector_size': self.vector_size,
            'device': self.device,
            'trained': self.trained,
            'model': self.model.model_name if self.model else None,
            "weight_path": self.model.weight if self.model else None,
            'redis_data': redis_data
        }
        json_path = f"{self.filename}.json"
        print(f"Saving database metadata and Redis data to {json_path}")
        with open(json_path, 'w') as f:
            json.dump(payload, f)
        print(f"Saved FAISS index, metadata and Redis data to disk.")