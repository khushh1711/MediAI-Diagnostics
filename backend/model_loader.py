import os
import torch
import torch.nn as nn
from torchvision import models
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

# 1. X-Ray Model
print("Loading X-Ray model...")
xray_model = models.resnet50(weights=None)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_path = os.path.join(MODELS_DIR, 'image_model.pth')
if os.path.exists(xray_path):
    xray_model.load_state_dict(torch.load(xray_path, map_location=device))
    print("X-Ray model ready! [OK]")
else:
    print(f"[WARNING]: {xray_path} not found. Running with uninitialized weights.")
xray_model = xray_model.to(device)
xray_model.eval()

# 2. Vitals Model
class VitalsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.network(x)

print("Loading Vitals model...")
vitals_model = VitalsModel().to(device)
vitals_path = os.path.join(MODELS_DIR, 'vitals_model.pth')
if os.path.exists(vitals_path):
    vitals_model.load_state_dict(torch.load(vitals_path, map_location=device))
    print("Vitals model ready! [OK]")
else:
    print(f"[WARNING]: {vitals_path} not found. Running with uninitialized weights.")
vitals_model.eval()

# 3. Scaler
print("Loading scaler...")
scaler_path = os.path.join(MODELS_DIR, 'vitals_scaler.pkl')
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler ready! [OK]")
else:
    print(f"[WARNING]: {scaler_path} not found. Using fallback scaler.")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

# 4. Brain Model
print("Loading Brain Tumor model...")
brain_model = models.efficientnet_b3(weights=None)
brain_model.classifier[1] = nn.Linear(brain_model.classifier[1].in_features, 4)
brain_path = os.path.join(MODELS_DIR, 'brain_tumor_model.pth')
if os.path.exists(brain_path):
    brain_model.load_state_dict(torch.load(brain_path, map_location=device))
    print("Brain Tumor model ready! [OK]")
else:
    print(f"[WARNING]: {brain_path} not found. Running with uninitialized weights.")
brain_model = brain_model.to(device)
brain_model.eval()