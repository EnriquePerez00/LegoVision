import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(data_dir, output_model, epochs=10, batch_size=32):
    device = get_device()
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando entrenamiento en: {device}")

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    print(f"[{time.strftime('%H:%M:%S')}] Dataset cargado: {len(dataset)} imágenes, {len(dataset.classes)} clases.")
    
    # Save classes mapping for inference
    classes_file = output_model + ".classes.txt"
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    with open(classes_file, "w") as f:
        for c in dataset.classes:
            f.write(f"{c}\n")
    print(f"Guardadas {len(dataset.classes)} clases en {classes_file}")
    
    num_classes = len(dataset.classes)
    model = timm.create_model('efficientnetv2_rw_s', pretrained=True, num_classes=num_classes)

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        t0 = time.time()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {running_loss/(batch_idx+1):.4f} | Acc: {100.*correct/total:.2f}%")
        
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} completada en {epoch_time:.1f}s | Loss: {running_loss/len(dataloader):.4f} | Acc: {100.*correct/total:.2f}%")
        
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"Modelo guardado en {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_model(args.data_dir, args.output_model, args.epochs)

