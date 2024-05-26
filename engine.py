import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(epoch, model, optimizer, loss_fn, train_loader: DataLoader, device, transform=None, lr_scheduler=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        if transform is not None:
            inputs = transform(inputs)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix(loss=total_loss/(batch_idx+1), accuracy=100.*total_correct/total_samples)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * total_correct / total_samples
    print(f"Training Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return avg_loss

def valid_one_epoch(model, loss_fn, optimizer, valid_loader: DataLoader, device, transform=None):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    preds = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(valid_loader, desc="Validating")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if transform:
                inputs = transform(inputs)
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.cpu().numpy())

            pbar.set_postfix(loss=total_loss/(batch_idx+1), accuracy=100.*total_correct/total_samples)

    avg_loss = total_loss / len(valid_loader)
    accuracy = 100. * total_correct / total_samples
    print(f"Validation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return accuracy, avg_loss, preds, labels
