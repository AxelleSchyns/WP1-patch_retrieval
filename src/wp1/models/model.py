from wp1.models.arch import *
REPO_DIR = '/home/axelle/Documents/Doctorat/Main research/dinov3' 

import torch

def map_dinov3_to_hf(checkpoint_path, model):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("teacher", ckpt)
    if "model" in sd and isinstance(sd["model"], dict) and not any(
            k.startswith(("backbone", "blocks", "cls_token")) for k in sd):
        sd = sd["model"]

    use_k_bias = model.config.key_bias
    hf = {}
    for k, v in sd.items():
        k = k.removeprefix("backbone.")          # your ckpts have it, official ones don't
        if k.startswith(("dino_head", "ibot_head", "rope_embed")) or "bias_mask" in k:
            continue
        if k.startswith("patch_embed.proj"):
            hf[k.replace("patch_embed.proj", "embeddings.patch_embeddings")] = v
        elif k == "cls_token":
            hf["embeddings.cls_token"] = v
        elif k == "mask_token":
            hf["embeddings.mask_token"] = v.unsqueeze(0) if v.ndim == 2 else v
        elif k in ("storage_tokens", "register_tokens"):
            hf["embeddings.register_tokens"] = v.unsqueeze(0) if v.ndim == 2 else v
        elif k.startswith("blocks."):
            nk = (k.replace("blocks.", "model.layer.", 1)
                   .replace("attn.proj", "attention.o_proj")
                   .replace("ls1.gamma", "layer_scale1.lambda1")
                   .replace("ls2.gamma", "layer_scale2.lambda1")
                   .replace("mlp.fc1", "mlp.up_proj")
                   .replace("mlp.fc2", "mlp.down_proj"))
            if ".attn.qkv." in k:
                dim = v.shape[0] // 3
                base, suffix = nk.replace(".attn.qkv", ".attention").rsplit(".", 1)
                hf[f"{base}.q_proj.{suffix}"] = v[:dim]
                hf[f"{base}.v_proj.{suffix}"] = v[2 * dim:]
                if suffix == "weight" or use_k_bias:
                    hf[f"{base}.k_proj.{suffix}"] = v[dim:2 * dim]
                continue
            hf[nk] = v
        elif k.startswith("norm."):
            hf[k] = v
    return model.load_state_dict(hf, strict=False)

def map_deit_weights_to_hf(state_dict, model):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remap modern Hugging Face structure to match the model wrapper's expected keys
        new_key = k
        new_key = new_key.replace("deit.encoder.layer.", "deit.layers.")
        new_key = new_key.replace(".attention.attention.query.", ".attention.q_proj.")
        new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
        new_key = new_key.replace(".attention.attention.value.", ".attention.v_proj.")
        new_key = new_key.replace(".attention.output.dense.", ".attention.o_proj.")
        new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
        new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        
        new_state_dict[new_key] = v
    return model.load_state_dict(new_state_dict, strict=True)

def loading_weights(model, model_name, weight, device):
        if "dinov3" in model_name:
            checkpoint_path = weight
            msg = map_dinov3_to_hf(checkpoint_path, model)
            print(f"Mapped Load Result: {msg}")
        elif "dino" in model_name or "ibot" in model_name or "cdpath" in model_name:
            model.load_weights(weight)
            model = model.model
        elif "byol" in model_name:
            try:
                model.load_state_dict(torch.load(weight)["state_dict"])
            except:
                model = BYOL(1000).to(device=device)
                model.load_state_dict(torch.load(weight)["state_dict"])
        elif model_name == "ret_ccl":
            model.fc = nn.Identity()
            model.load_state_dict(torch.load(weight), strict=True)
        elif model_name == "ctranspath":
            model.head = nn.Identity()
            model.load_state_dict(torch.load(weight)['model'])
        elif model_name == "uni_v1":
            model.load_state_dict(torch.load(weight))
        elif model_name == "virchow_v2":
            model.load_state_dict(torch.load(weight, weights_only=False).state_dict())
        elif model_name == "deit":
            checkpoint = torch.load(weight)
            msg = map_deit_weights_to_hf(checkpoint["model_state_dict"], model)
            print(f"Mapped Load Result: {msg}")
        elif model_name in ["phikon_v1", "phikon_v2", "hoptim_v0", "uni_v2", "hoptim_v1"]:
            pass
        else:
            try:
                checkpoint = torch.load(weight, weights_only=False)
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                msg = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights successfully: {msg}")
            except Exception as e:
                print("Error with the loading of the model's weights: ", e)
                print("Exiting...")
                sys.exit(-1)

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
        self.num_features = model_object[1]
        self.model = model_object[0].to(device=device)

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

            elif self.model_name in {"phikon_v1", "phikon_v2"}:
                outputs = self.model(image)
                out = outputs.last_hidden_state[:, 0, :]

            elif self.model_name in {"hoptim_v0", "hoptim_v1"} and self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(image)

            elif self.model_name == "virchow_v2":
                output = self.model(image)
                class_token = output[:, 0]
                patch_tokens = output[:, 5:]
                out = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            elif "dinov3" in self.model_name:
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