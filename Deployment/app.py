import io
import os
from typing import List, Any
from PIL import Image
import torch
import torchvision.transforms as T
from flask import Flask, request, jsonify

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_cifar10.pth")
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

app = Flask(__name__)

# Lazy-load model so startup is fast
_model = None
_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def _build_fallback_model(num_classes: int = 10) -> torch.nn.Module:
    # Minimal CNN fallback (only used if a state_dict is provided without full model object)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 8 * 8, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, num_classes)
    )


def load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = torch.load(MODEL_PATH, map_location=device)
    # Case 1: directly a torch.nn.Module
    if isinstance(loaded, torch.nn.Module):
        _model = loaded.to(device).eval()
        return _model
    # Case 2: dictionary that might contain 'model_state' or direct state_dict
    if isinstance(loaded, dict):
        # Heuristics
        possible_keys = ["state_dict", "model_state", "model_state_dict"]
        state = None
        for k in possible_keys:
            if k in loaded:
                state = loaded[k]
                break
        if state is None:
            # maybe it's already a state_dict
            if all(isinstance(v, torch.Tensor) for v in loaded.values()):
                state = loaded
        if state is not None:
            model = _build_fallback_model(len(CLASS_NAMES))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[warn] Missing keys when loading model: {missing}")
            if unexpected:
                print(f"[warn] Unexpected keys when loading model: {unexpected}")
            _model = model.to(device).eval()
            return _model
    raise RuntimeError("Unsupported model file format. Save the whole model with torch.save(model) or a dict with a 'state_dict' key.")


def prepare_image(img: Image.Image):
    return _transform(img.convert("RGB")).unsqueeze(0)


@app.route("/health", methods=["GET"])  # simple health check
def health():
    return {"status": "ok"}


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    tensor = prepare_image(img)
    model = load_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    # Sort probabilities descending
    class_probs = sorted([(CLASS_NAMES[i], float(p)) for i, p in enumerate(probs)], key=lambda x: x[1], reverse=True)

    # Render HTML result page with sorted % and back button
    html = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8' />",
        "<title>Prediction Result</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:480px;margin:40px auto;padding:0 1rem;color:#222}h1{font-size:1.3rem}table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{padding:.5rem;text-align:left;border-bottom:1px solid #eee}tr:first-child{background:#e6f7ff;font-weight:bold}tr:hover{background:#f5f5f5}button{margin-top:1.5rem;cursor:pointer;background:#0366d6;color:#fff;border:none;padding:.6rem 1.2rem;border-radius:4px;font-size:.95rem}img{max-width:120px;display:block;margin:1rem auto 0 auto;border-radius:8px;box-shadow:0 2px 8px #0001}</style>",
        "</head><body>",
        "<h1>Prediction Result</h1>"
    ]
    # Show uploaded image preview (optional, if you want to display it)
    # html.append(f"<img src='data:image/png;base64,{base64.b64encode(img_bytes).decode()}' alt='Uploaded image' />")
    html.append("<table><tr><th>Class</th><th>Probability (%)</th></tr>")
    for cname, prob in class_probs:
        html.append(f"<tr><td>{cname}</td><td>{prob*100:.2f}%</td></tr>")
    html.append("</table>")
    html.append(f"<p><b>Top prediction:</b> {class_probs[0][0]} ({class_probs[0][1]*100:.2f}%)</p>")
    html.append("<form action='/' method='get'><button type='submit'>&larr; Back / Upload another image</button></form>")
    html.append("</body></html>")
    return "".join(html)


@app.route("/", methods=["GET"])
def upload_form():
    """Lightweight HTML form for manual testing in a browser."""
    return (
        """<!DOCTYPE html><html lang='en'>\n<head>\n<meta charset='UTF-8' />\n<title>CIFAR-10 Predictor</title>\n<style>body{font-family:system-ui,Arial,sans-serif;max-width:640px;margin:40px auto;padding:0 1rem;color:#222}h1{font-size:1.4rem;margin-bottom:.5rem}form{border:1px solid #ccc;padding:1rem;border-radius:8px;background:#fafafa}#preview{max-width:160px;margin-top:.5rem;display:none;border:1px solid #ddd;padding:4px;border-radius:4px}button{cursor:pointer;background:#0366d6;color:#fff;border:none;padding:.6rem 1rem;border-radius:4px;font-size:.95rem}button:disabled{opacity:.5;cursor:not-allowed}code{background:#eee;padding:2px 4px;border-radius:4px;font-size:.85rem;display:inline-block;margin-top:4px}#result pre{background:#111;color:#0f0;padding:.75rem;border-radius:6px;overflow:auto;font-size:.8rem}footer{margin-top:2rem;font-size:.7rem;color:#666}</style>\n</head><body>\n<h1>CIFAR-10 Image Prediction</h1>\n<p>Select an image (32x32 or larger) and submit to get class probabilities.</p>\n<form id='predict-form' enctype='multipart/form-data' method='post' action='/predict'>\n  <input type='file' id='file' name='file' accept='image/*' required />\n  <div><img id='preview' alt='preview'></div>\n  <div style='margin-top:1rem;display:flex;gap:.5rem;align-items:center'>\n    <button type='submit' id='btn'>Predict</button>\n    <span id='status'></span>\n  </div>\n</form>\n<section id='result'></section>\n<footer>Endpoint: <code>/predict</code> | Health: <code>/health</code></footer>\n<script>\nconst form=document.getElementById('predict-form');\nconst fileInput=document.getElementById('file');\nconst preview=document.getElementById('preview');\nfileInput.addEventListener('change',()=>{const f=fileInput.files[0];if(!f){preview.style.display='none';return;}const url=URL.createObjectURL(f);preview.src=url;preview.style.display='block';});\n</script>\n</body></html>"""
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
