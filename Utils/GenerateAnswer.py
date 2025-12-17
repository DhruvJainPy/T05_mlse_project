import os
import requests
from pathlib import Path
import torch
import open_clip
from PIL import Image
from dotenv import load_dotenv

# ============================
# ENV + NVIDIA CONFIG
# ============================
load_dotenv()

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_API_KEY = os.getenv("LLAMA_API")
LLAMA_INSTRUCT_MODEL = "meta/llama-4-scout-17b-16e-instruct"

if NVIDIA_API_KEY is None:
    raise RuntimeError("Environment variable LLAMA_API is not set.")


# ============================
# 1. Device
# ============================
device = "cpu" 
print("Using device:", device)

# ============================
# 2. Load CLIP model
# ============================
model_name = "ViT-B-32"
pretrained = "openai"

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)
tokenizer = open_clip.get_tokenizer(model_name)

model = model.to(device)
model.eval()

print("Loaded CLIP model:", model_name, "pretrained:", pretrained)

# ============================
# 3. Load saved embeddings
# ============================

BASE_DIR = Path(__file__).resolve().parent
EMB_PATH = BASE_DIR / "Clip_Embeddings" / "panel_clip_embeddings.pt"

if not EMB_PATH.exists():
    raise FileNotFoundError(f"Embeddings file not found: {EMB_PATH.resolve()}")

bundle = torch.load(EMB_PATH, map_location="cpu")

panel_ids      = bundle["panel_ids"]
image_paths    = bundle["image_paths"]
texts          = bundle["texts"]
image_features = bundle["image_features"]   
text_features  = bundle["text_features"]    

print("Loaded features:")
print("  image_features:", image_features.shape)
print("  text_features :", text_features.shape)

image_features = image_features / image_features.norm(dim = -1, keepdim=True)
text_features  = text_features  / text_features.norm(dim = -1, keepdim=True)

image_features = image_features.to("cpu")
text_features  = text_features.to("cpu")

panel_meta = {
    pid: {"panel_id": pid, "image_path": ipath, "text": txt}
    for pid, ipath, txt in zip(panel_ids, image_paths, texts)
}


# ==================================================
# 4. Llama-4 instruct as TEXT–TEXT cross-encoder
# ==================================================
def score_with_llama_instruct(anchor_text: str, candidate_texts):
    """
    Use meta/llama-4-scout-17b-16e-instruct as a text-text cross encoder.

    anchor_text: text summarizing the image (e.g. top CLIP caption)
    candidate_texts: list of captions from CLIP retrieval

    returns: list[float] scores in [0, 1]
    """
    scores = []

    for cand in candidate_texts:
        prompt = f"""
You are a similarity scoring model.

I will give you:
- A QUERY description
- A CANDIDATE description

You must respond with ONLY a single number between 0 and 1
indicating how semantically similar the CANDIDATE is to the QUERY.
0 = completely unrelated, 1 = identical meaning.
No explanation, no extra text.

QUERY:
{anchor_text}

CANDIDATE:
{cand}
"""

        payload = {
            "model": LLAMA_INSTRUCT_MODEL,
            "messages": [
                {"role": "user", "content": prompt.strip()}
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        resp = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        
        try:
            score = float(raw.split()[0])
        except Exception:
            score = 0.0

        # clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        scores.append(score)

    return scores


# ==================================================
# 5. IMAGE -> TEXT with CLIP + Llama re-ranking
# ==================================================
@torch.no_grad()
def search_texts_by_image(
    image_path: str,
    k: int = 10,
    k_clip: int = 30,
    use_cross_encoder: bool = True,
):
    """
    IMAGE -> TEXT

    Step 1: CLIP retrieves top-k_clip text candidates for the image (fast, over full DB).
    Step 2: Llama-4 instruct re-scores candidates as text-text similarity
            vs the best CLIP caption (pseudo description of the image).
    Step 3: Return top-k by Llama score.
    """
    # ---- Step 1: CLIP retrieval ----
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    i_feat = model.encode_image(img_tensor)
    i_feat = i_feat / i_feat.norm(dim=-1, keepdim=True)
    i_feat_cpu = i_feat.to("cpu")

    sims = torch.matmul(i_feat_cpu, text_features.t())[0]  # [N]

    # prune to k_clip candidates
    k_clip = min(k_clip, sims.shape[0])
    clip_scores, clip_indices = torch.topk(sims, k_clip)

    candidates = []
    for rank, (score, idx) in enumerate(zip(clip_scores, clip_indices), start=1):
        idx = idx.item()
        candidates.append({
            "clip_rank": rank,
            "clip_score": float(score.item()),
            "panel_id": str(panel_ids[idx]),
            "image_path": str(image_paths[idx]),
            "db_text": str(texts[idx]),
            "metadata": panel_meta.get(panel_ids[idx], {}),
        })

    if not use_cross_encoder or len(candidates) == 0:
        candidates = candidates[:k]
        for i, c in enumerate(candidates, start=1):
            c["rank"] = i
            c["score"] = c["clip_score"]
        return candidates

    # ---- Step 2: Llama-4 instruct text-text re-ranking ----
    anchor_text = candidates[0]["db_text"]
    cand_texts = [c["db_text"] for c in candidates]

    ce_scores = score_with_llama_instruct(anchor_text, cand_texts)

    for c, s in zip(candidates, ce_scores):
        c["ce_score"] = float(s)

    candidates.sort(key=lambda x: x["ce_score"], reverse=True)

    k = min(k, len(candidates))
    results = []
    for rank in range(k):
        c = candidates[rank]
        results.append({
            "rank": rank + 1,
            "score": c["ce_score"],       
            "clip_score": c["clip_score"],
            "panel_id": c["panel_id"],
            "image_path": c["image_path"],
            "db_text": c["db_text"],
            "metadata": c["metadata"],
        })
    return results


# ==================================================
# 6. TEXT -> IMAGE with CLIP
# ==================================================
@torch.no_grad()
def search_images_by_text(query: str, k: int = 5):
    tokens = tokenizer([query]).to(device)
    q_feat = model.encode_text(tokens)
    q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)

    q_feat_cpu = q_feat.to("cpu")
    sims = torch.matmul(q_feat_cpu, image_features.t())[0]

    k = min(k, sims.shape[0])
    topk_scores, topk_indices = torch.topk(sims, k)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores, topk_indices), start=1):
        idx = idx.item()
        results.append({
            "rank": rank,
            "score": float(score.item()),
            "panel_id": str(panel_ids[idx]),
            "image_path": str(image_paths[idx]),
            # "db_text": str(texts[idx]),
            "metadata": panel_meta.get(panel_ids[idx], {}),
        })
    return results


# ==================================================
# 7. UNIFIED ENTRY: pass text OR image path
# ==================================================
def multimodal_search(
    query: str,
    k: int = 5,
    k_clip: int = 30,
    use_cross_encoder: bool = True,
):
    """
    If `query` is an existing file path → treat as IMAGE and return TEXT results.
    Otherwise → treat as TEXT and return IMAGE results.
    """
    path = Path(query)
    if path.exists() and path.is_file():
        # IMAGE -> TEXT
        print(f"\n[MODE] IMAGE -> TEXT   (file: {path})")
        return "image_to_text", search_texts_by_image(str(path), k = k, k_clip = k_clip, use_cross_encoder = use_cross_encoder)
    else:
        # TEXT -> IMAGE
        print(f"\n[MODE] TEXT -> IMAGE   (query: {query})")
        return "text_to_image", search_images_by_text(query, k = k)

        
