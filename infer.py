import argparse, torch
from PIL import Image
from torchvision import transforms, models

# Class labels
classes = ["helmet","no_helmet"]

# Preprocessing (must match training)
tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_model(ckpt="models/resnet18_helmet.pt"):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model

def predict(path, model):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    args = parser.parse_args()

    model = load_model()
    label, prob = predict(args.image, model)
    print(f"{args.image} -> {label} (p={prob:.2f})")
