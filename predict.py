import torchxrayvision as xrv

print("Loading model...")

model = xrv.models.DenseNet(weights="densenet121-res224-all")

print("Model loaded!")

print("\nDiseases the model can predict:")

for pathology in model.pathologies:
    print("-", pathology)