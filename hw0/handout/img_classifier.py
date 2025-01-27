import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse
import wandb

img_size = (256,256)
num_labels = 3

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    transform_img = T.Compose([
        T.ToTensor(), 
        T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
        T.CenterCrop(img_size),  # Center crop to 256x256
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
        ])
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )

    validation_data = CsvImageDataset(csv_file='./data/img_val.csv', transform=transform_img)  

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)  # validation DataLoader 추가

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, validation_dataloader, test_dataloader

def log_first_batch(dataloader, dataname, model, class_names):
    model.eval() 
    with torch.no_grad():
        X, y = next(iter(dataloader))
        X, y = X.to(device), y.to(device)

        pred = model(X).argmax(1)

        images = [
            wandb.Image(
                img, 
                caption=f"{class_names[pred_label]} / {class_names[true_label]}"
            )
            for img, pred_label, true_label in zip(X.cpu(), pred.cpu().numpy(), y.cpu().numpy())
        ]

        # wandb 로깅
        wandb.log({f"{dataname} First Batch Predictions": images})

class NeuralNetwork(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()

        # Convolutional layers
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=4),  # (B, 3, 256, 256) -> (B, 128, 64, 64)
            nn.LayerNorm([128, 64, 64]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, padding=3),  # (B, 128, 64, 64)
            nn.LayerNorm([128, 64, 64]),
        
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),  # (B, 128, 64, 64) -> (B, 256, 64, 64)
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),  # (B, 256, 64, 64) -> (B, 128, 64, 64)

            nn.AvgPool2d(kernel_size=2),  # (B, 128, 64, 64) -> (B, 128, 32, 32)
            nn.Flatten(),  # (B, 128, 32, 32) -> (B, 128 * 32 * 32)
            nn.Linear(128 * 32 * 32, num_labels)  # (B, 3)
        )

    def forward(self, x):
        logits = self.conv_stack(x)  
        return logits



def train_one_epoch(dataloader, model, loss_fn, optimizer, t, examples_seen):
    size = len(dataloader.dataset)
    total_loss = 0
    correct = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = len(y)
        avg_loss = loss.item() / batch_size
        total_loss += loss.item()
        examples_seen += batch_size

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        wandb.log({
            "Batch Loss": avg_loss,
            "Examples Seen": examples_seen,
            "Epoch": t
        }, step=examples_seen)


        if batch % 10 == 0:
            print(f"Train batch avg loss = {avg_loss:>7f}  [{examples_seen:>5d}/{size:>5d}]")
    
    # Calculate average loss for the epoch
    avg_train_loss = total_loss / size
    train_accuracy = correct / size  # Accuracy as the ratio of correct predictions to total examples
    return avg_train_loss, train_accuracy, examples_seen
        
def evaluate(dataloader, dataname, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss = 0
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            #avg_loss += loss_fn(pred, y).item()
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    #correct /= size
    #print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")

    avg_loss = total_loss / size
    accuracy = correct / size
    return avg_loss, accuracy
    
def main(n_epochs, batch_size, learning_rate):
    wandb.init(project="img_classification_project", name="neural-the-narwhal")

    print(f"Using {device} device")
    train_dataloader, validation_dataloader, test_dataloader = get_data(batch_size)
    
    model = NeuralNetwork().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    examples_seen = 0

    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_loss, train_acc, examples_seen = train_one_epoch(train_dataloader, model, loss_fn, optimizer, t, examples_seen)
        val_loss, val_acc = evaluate(validation_dataloader, "validation", model, loss_fn)
        test_loss, test_acc = evaluate(test_dataloader, "test", model, loss_fn)

        if t == n_epochs - 1:
            class_names = ["parrot", "narwhal", "axolotl"]
            log_first_batch(train_dataloader, "Train", model, class_names)
            log_first_batch(validation_dataloader, "Validation", model, class_names)
            log_first_batch(test_dataloader, "Test", model, class_names)

        # Log epoch-level metrics to WandB
        wandb.log({
            "Train Loss (epoch avg)": train_loss,
            "Train Accuracy": train_acc,  # Log train accuracy
            "Validation Loss (epoch avg)": val_loss,
            "Validation Accuracy": val_acc,
            "Test Loss (epoch avg)": test_loss,  
            "Test Accuracy": test_acc, 
            "Epoch": t
        })

        print(f"Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_acc:.2%}")

    print("Done!")

    # Save the model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

    # Load the model (just for the sake of example)
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=20, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)

    args = parser.parse_args()
        
    main(args.n_epochs, args.batch_size, args.learning_rate)