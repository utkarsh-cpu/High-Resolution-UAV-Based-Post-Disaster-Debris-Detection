"""Quick debug script: check what Florence-2 actually generates."""
import torch
from PIL import Image
from hurricane_debris.models.florence2 import load_florence_processor
from hurricane_debris.models.cascade import _patch_florence2_config
from peft import PeftModel
from transformers import AutoModelForCausalLM
import json

MODEL_DIR = "./models/florence2_debris"
TEST_IMG = "./datasets/rescuenet/RescueNet/test/test-org-img/10849.jpg"

# Load processor
processor = load_florence_processor(MODEL_DIR)

# Patch for transformers 5.x
_patch_florence2_config()

# Load base + adapter
with open(f"{MODEL_DIR}/adapter_config.json") as f:
    acfg = json.load(f)
base_id = acfg["base_model_name_or_path"]
print(f"Base model: {base_id}")

base_model = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.float32, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, MODEL_DIR).merge_and_unload()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

image = Image.open(TEST_IMG).convert("RGB")
print(f"Image size: {image.size}")

# Test 1: bare task token (how training was done)
prompt1 = "<OPEN_VOCABULARY_DETECTION>"
inputs1 = processor(text=prompt1, images=image, return_tensors="pt").to(device)
print(f"\n--- Prompt: {prompt1!r} ---")
print(f"input_ids shape: {inputs1['input_ids'].shape}")
print(f"input_ids: {inputs1['input_ids']}")

with torch.no_grad():
    gen1 = model.generate(
        input_ids=inputs1["input_ids"],
        pixel_values=inputs1["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        use_cache=False,
    )
text1 = processor.batch_decode(gen1, skip_special_tokens=False)[0]
print(f"Raw output: {text1[:1000]}")

parsed1 = processor.post_process_generation(
    text1, task="<OPEN_VOCABULARY_DETECTION>", image_size=image.size
)
print(f"Parsed: {parsed1}")

# Test 2: with query (how cascade was calling it before fix)
prompt2 = "<OPEN_VOCABULARY_DETECTION>debris, damaged building, flooded area, downed tree, damaged road, vehicle wreckage"
inputs2 = processor(text=prompt2, images=image, return_tensors="pt").to(device)
print(f"\n--- Prompt: {prompt2!r} ---")
print(f"input_ids shape: {inputs2['input_ids'].shape}")

with torch.no_grad():
    gen2 = model.generate(
        input_ids=inputs2["input_ids"],
        pixel_values=inputs2["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        use_cache=False,
    )
text2 = processor.batch_decode(gen2, skip_special_tokens=False)[0]
print(f"Raw output: {text2[:1000]}")

parsed2 = processor.post_process_generation(
    text2, task="<OPEN_VOCABULARY_DETECTION>", image_size=image.size
)
print(f"Parsed: {parsed2}")

# Test 3: Try <OD> task token (standard Florence-2 object detection)
prompt3 = "<OD>"
inputs3 = processor(text=prompt3, images=image, return_tensors="pt").to(device)
print(f"\n--- Prompt: {prompt3!r} ---")
with torch.no_grad():
    gen3 = model.generate(
        input_ids=inputs3["input_ids"],
        pixel_values=inputs3["pixel_values"],
        max_new_tokens=512,
        num_beams=3,
        use_cache=False,
    )
text3 = processor.batch_decode(gen3, skip_special_tokens=False)[0]
print(f"Raw output: {text3[:1000]}")
