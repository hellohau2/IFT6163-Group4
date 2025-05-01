"""
build_centroids.py  <root_folder>
Scans every sub-folder under <root_folder>, encodes images with CLIP
(ViT-L/14-quickgelu), averages them, L2-normalises, and saves
centroids.pt  (a dict {class: tensor(1,768)}).
"""
import os, glob, torch, argparse, open_clip
from PIL import Image

def centroid(folder, preprocess, device, model):
    imgs = [preprocess(Image.open(p)) for p in glob.glob(f"{folder}/*")]
    if not imgs: return None
    batch = torch.stack(imgs).to(device)
    with torch.no_grad():
        f = model.encode_image(batch); f = f / f.norm(dim=-1, keepdim=True)
    return f.mean(0, keepdim=True).cpu()

def main(root):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14-quickgelu", pretrained="openai", device=device)
    model.eval()

    cents = {}
    for sub in os.listdir(root):
        path = os.path.join(root, sub)
        if not os.path.isdir(path): continue
        c = centroid(path, preprocess, device, model)
        if c is not None:
            cents[sub] = c
            print(f"{sub:14s} {len(os.listdir(path))} imgs")
    torch.save(cents, "centroids.pt")
    print("Saved centroids.pt with", len(cents), "classes")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with class sub-folders")
    main(ap.parse_args().root)