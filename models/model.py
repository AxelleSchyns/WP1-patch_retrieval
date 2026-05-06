from models.arch import *
from training.loss import MarginLoss, ProxyNCA_prob, NormSoftmax, SoftTriple
REPO_DIR = '/home/labsig/Documents/Axelle/Main research/ext_models/dinov3-main' 

def map_dinov3_to_hf(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint.get("teacher", checkpoint)
    hf_state_dict = {}
    
    for k, v in state_dict.items():
        new_k = k
        # 1. Handle Embeddings
        if "patch_embed.proj" in k:
            new_k = k.replace("backbone.patch_embed.proj", "embeddings.patch_embeddings")
        elif "cls_token" in k:
            new_k = "embeddings.cls_token"
        elif "mask_token" in k:
            new_k = "embeddings.mask_token"
        
        # 2. Handle Blocks/Layers
        if "backbone.blocks." in k:
            # Convert 'backbone.blocks.0...' to 'model.layer.0...'
            new_k = k.replace("backbone.blocks.", "model.layer.")
            
            # Map Internal Layer Components
            new_k = new_k.replace("attn.proj", "attention.o_proj")
            new_k = new_k.replace("ls1.gamma", "layer_scale1.lambda1")
            new_k = new_k.replace("ls2.gamma", "layer_scale2.lambda1")
            new_k = new_k.replace("mlp.fc1", "mlp.up_proj")
            new_k = new_k.replace("mlp.fc2", "mlp.down_proj")

            # Special handling for QKV split
            if "attn.qkv" in k:
                dim = v.shape[0] // 3
                prefix = new_k.replace("attention.qkv", "attention")
                hf_state_dict[prefix + ".q_proj" + k[-7:]] = v[:dim]
                hf_state_dict[prefix + ".k_proj" + k[-7:]] = v[dim : 2 * dim]
                hf_state_dict[prefix + ".v_proj" + k[-7:]] = v[2 * dim :]
                continue

        # 3. Handle Final Norm
        if "backbone.norm" in k:
            new_k = k.replace("backbone.norm", "norm")

        hf_state_dict[new_k] = v

    return model.load_state_dict(hf_state_dict, strict=False)

def map_dinov3_large_to_hf(checkpoint_path, model):
    """
    Maps DINOv3 research checkpoint keys to Hugging Face AutoModel keys.
    Compatible with ViT-S, ViT-L, and models with/without register tokens.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract the state_dict from the 'teacher' key if it exists
    state_dict = checkpoint.get("teacher", checkpoint)
    if "model" in state_dict and not any(k.startswith("backbone") for k in state_dict.keys()):
        state_dict = state_dict["model"]

    hf_state_dict = {}
    
    for k, v in state_dict.items():
        # 1. Handle Embeddings and Special Tokens
        if "backbone.patch_embed.proj" in k:
            new_k = k.replace("backbone.patch_embed.proj", "embeddings.patch_embeddings")
        elif "cls_token" in k:
            new_k = "embeddings.cls_token"
        elif "mask_token" in k:
            new_k = "embeddings.mask_token"
            if v.ndim == 2: v = v.unsqueeze(0) # Fix [1, dim] -> [1, 1, dim]
        elif "register_tokens" in k:
            new_k = "embeddings.register_tokens"
            if v.ndim == 2: v = v.unsqueeze(0) # Fix [N, dim] -> [1, N, dim]
        
        # 2. Handle Transformer Blocks
        elif "backbone.blocks." in k:
            # Basic mapping for block components
            new_k = k.replace("backbone.blocks.", "model.layer.")
            new_k = new_k.replace("attn.proj", "attention.o_proj")
            new_k = new_k.replace("ls1.gamma", "layer_scale1.lambda1")
            new_k = new_k.replace("ls2.gamma", "layer_scale2.lambda1")
            new_k = new_k.replace("mlp.fc1", "mlp.up_proj")
            new_k = new_k.replace("mlp.fc2", "mlp.down_proj")

            # Special Handling for QKV Split
            if "attn.qkv" in k:
                # Calculate dimension for splitting
                total_dim = v.shape[0]
                single_dim = total_dim // 3
                
                # Determine weight vs bias suffix
                suffix = ".weight" if k.endswith(".weight") else ".bias"
                
                # Clean the base path: backbone.blocks.0.attn.qkv.weight -> model.layer.0.attention
                base_path = new_k.replace(".attn.qkv", ".attention").replace(suffix, "")
                
                # Map to the three separate HF projections
                hf_state_dict[f"{base_path}.q_proj{suffix}"] = v[:single_dim]
                hf_state_dict[f"{base_path}.k_proj{suffix}"] = v[single_dim : 2 * single_dim]
                hf_state_dict[f"{base_path}.v_proj{suffix}"] = v[2 * single_dim :]
                continue # Skip the default assignment below

        # 3. Handle Final Layer Norm
        elif "backbone.norm" in k:
            new_k = k.replace("backbone.norm", "norm")
        
        else:
            # Skip heads (dino_head, ibot_head) or non-backbone keys
            continue

        # Add the mapped key to our new dictionary
        hf_state_dict[new_k] = v

    # Load into model
    msg = model.load_state_dict(hf_state_dict, strict=False)
    return msg


def loading_weights(model, model_name, weight, device):
        print(model_name, weight)
        print("HEY\n", flush=True)
        if model_name in ['dino_vit', 'dino_resnet','dino_tiny', "cdpath", "ibot_vits" , "ibot_vitb"]:
                model.load_weights(weight)
                model = model.model

        elif model_name == "byol_light" :
            try:
                model.load_state_dict(torch.load(weight)["state_dict"])
            except:
                model = BYOL(1000).to(device=device)
                model.load_state_dict(torch.load(weight)["state_dict"])

        elif model_name == "ret_ccl":
            pretext_model = torch.load(weight)
            model.fc = nn.Identity()
            model.load_state_dict(pretext_model, strict=True)

        elif model_name in ["phikon", "phikon2", "hoptim", "uni2",  "hoptim1", "virchow2"]: 
            pass
        elif model_name == "dinov3":
            # Load the checkpoint
            checkpoint_path = REPO_DIR + '/finetuned_dino_L/eval/training_124999/teacher_checkpoint.pth'
            #checkpoint_path = REPO_DIR + '/teacher_checkpoint.pth'
            # Usage
            msg = map_dinov3_large_to_hf(checkpoint_path, model)
            print(f"Mapped Load Result: {msg}")
        elif model_name == "ctranspath":
            model.head = nn.Identity()
            model.load_state_dict(torch.load(weight)['model'])

        elif model_name == "uni" or model_name == "virchow2":
            model.load_state_dict(torch.load(weight))
        else:
            try:
                model.load_state_dict(torch.load(weight))
            except Exception as e:
                try:
                    checkpoint = torch.load(weight)
                    print(checkpoint.keys())
                    model.load_state_dict(checkpoint['model_state_dict'])
                    #self.model.load_state_dict(checkpoint)
                except Exception as e:
                    print("Error with the loading of the model's weights: ", e) 
                    print("Exiting...")
                    exit(-1)

        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        return model

class Model(nn.Module):
    def __init__(self, model_name='resnet', weight='weights', device='cuda:0'):
        super().__init__()
        self.weight = weight
        self.model_name = model_name
        self.device = device
        model_object = load_arch(model_name)
        self.model = model_object[0]
        self.num_features = model_object[1]
        self.model = self.model.to(device=device)
        self.model = loading_weights(self.model, self.model_name, self.weight, self.device)

    def encode(self, image):
        image = image.to(device=self.device, non_blocking=True).reshape(-1, 3, 224, 224)
        with torch.inference_mode():
            if self.model_name == "resnet":
                out = self.model(image)

            elif self.model_name == "deit":
                out = self.model(image)
                out = out.logits

            elif self.model_name == "cdpath":
                image = scale_generator(image, 224, 1, 112, rescale_size=224)
                out = self.model.encode(image)

            elif self.model_name in {"phikon", "phikon2"}:
                outputs = self.model(image)
                out = outputs.last_hidden_state[:, 0, :]

            elif self.model_name in {"hoptim", "hoptim1"} and self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(image)

            elif self.model_name in {"hoptim", "hoptim1"}:  
                out = self.model(image)

            elif self.model_name == "virchow2":
                output = self.model(image)
                class_token = output[:, 0]
                patch_tokens = output[:, 5:]
                out = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            elif self.model_name == 'dinov3':
                with torch.inference_mode():
                    outputs = self.model(image)
       
                out = outputs.pooler_output


            else:
                out = self.model(image)
                if not isinstance(out, torch.Tensor):
                    out = out.logits

        # 🔹 Normalize for similarity search
        out = F.normalize(out, p=2, dim=-1)
        return out