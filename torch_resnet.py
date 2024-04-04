from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os

# Define transforms to preprocess the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit ResNet50 input size
    transforms.ToTensor(),  # Convert PIL images to tensors
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(freeze_layers=False):
    # Define ResNet50 model
    model = resnet50(weights=None)  # Load pre-trained ResNet50 weights
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Change the last fully connected layer to output 10 classes for MNIST
    model = model.to(device)
    return model

def train_model(model, train_dataset):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(10):  # Example of 5 epochs, you can adjust as needed
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

def evaluate_model(model, test_dataset):
    # Optionally, evaluate the model
    model.eval()
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %f %%' % (accuracy))
    return accuracy

def pretraining_model(sources, location_path):
    for source in sources:
        model = create_model()
        train_path = Path(location_path, "sources", source)
        train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
        
        train_model(model, train_dataset)
        torch.save(model.state_dict(), 'weights\\' + source + '_weights.pth')
        
        
        test_path = Path(location_path, "tests", "MNIST")
        test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
        evaluate_model(model, test_dataset)

def train_with_target(targets, location_path):
    
    for target in targets:
        exp_conditions = target.split("_")

        for i in range(10):
            print("Batch: ", i)
            model = create_model()
            model.load_state_dict(torch.load("weights\\" + exp_conditions[0] + "_" + exp_conditions[1] + "_weights.pth"))
            
            train_path = Path(location_path, "targets", target, "batch_" + str(i), "train") 
            train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
            train_model(model, train_dataset)

            test_path = Path(location_path, "tests", "SVHN")
            test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
            accuracy = evaluate_model(model, test_dataset)

            
            file1 = open("results.txt", "a")
            file1.write(f"{exp_conditions[0]}\t'{exp_conditions[1]}'\t{exp_conditions[2]}\t'{exp_conditions[3]}'\t{exp_conditions[4]}\t{accuracy}")
            file1.write("\n")
            file1.close()



def delete_empty_folders(path):
    # Iterate over the contents of the current directory
    for root, dirs, files in os.walk(path, topdown=False):
        for directory in dirs:
            # Construct the full path of the directory
            current_dir = os.path.join(root, directory)
            # If the directory is empty, remove it
            if not os.listdir(current_dir):
                print(f"Deleting empty directory: {current_dir}")
                os.rmdir(current_dir)

def list_folders(path):
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return []

    entries = os.listdir(path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(path, entry))]

    print(folders)

def main():
    print("Is CUDA available?", torch.cuda.is_available())
    location_path = f"C:\\Users\\Barnum\\Desktop\\experiments7\\"
    sources = ["MNIST_400", "MNIST_4000", "MNIST_40000"]

    targets = ['MNIST_40000_SVHN_1000_LinealFunction', 'MNIST_40000_SVHN_1000_NoneFunction', 'MNIST_40000_SVHN_1000_StepFunctionNegative', 'MNIST_40000_SVHN_1000_StepFunctionPositive', 'MNIST_40000_SVHN_100_LinealFunction', 'MNIST_40000_SVHN_100_NoneFunction', 'MNIST_40000_SVHN_100_StepFunctionNegative', 'MNIST_40000_SVHN_100_StepFunctionPositive', 'MNIST_40000_SVHN_4000_LinealFunction', 'MNIST_40000_SVHN_4000_NoneFunction', 'MNIST_40000_SVHN_4000_StepFunctionNegative', 'MNIST_40000_SVHN_4000_StepFunctionPositive', 'MNIST_40000_SVHN_500_LinealFunction', 'MNIST_40000_SVHN_500_NoneFunction', 'MNIST_40000_SVHN_500_StepFunctionNegative', 'MNIST_40000_SVHN_500_StepFunctionPositive', 'MNIST_4000_SVHN_1000_LinealFunction', 'MNIST_4000_SVHN_1000_NoneFunction', 'MNIST_4000_SVHN_1000_StepFunctionNegative', 'MNIST_4000_SVHN_1000_StepFunctionPositive', 'MNIST_4000_SVHN_100_LinealFunction', 'MNIST_4000_SVHN_100_NoneFunction', 'MNIST_4000_SVHN_100_StepFunctionNegative', 'MNIST_4000_SVHN_100_StepFunctionPositive', 'MNIST_4000_SVHN_4000_LinealFunction', 'MNIST_4000_SVHN_4000_NoneFunction', 'MNIST_4000_SVHN_4000_StepFunctionNegative', 'MNIST_4000_SVHN_4000_StepFunctionPositive', 'MNIST_4000_SVHN_500_LinealFunction', 'MNIST_4000_SVHN_500_NoneFunction', 'MNIST_4000_SVHN_500_StepFunctionNegative', 'MNIST_4000_SVHN_500_StepFunctionPositive', 'MNIST_400_SVHN_1000_LinealFunction', 'MNIST_400_SVHN_1000_NoneFunction', 'MNIST_400_SVHN_1000_StepFunctionNegative', 'MNIST_400_SVHN_1000_StepFunctionPositive', 'MNIST_400_SVHN_100_LinealFunction', 'MNIST_400_SVHN_100_NoneFunction', 'MNIST_400_SVHN_100_StepFunctionNegative', 'MNIST_400_SVHN_100_StepFunctionPositive', 'MNIST_400_SVHN_4000_LinealFunction', 'MNIST_400_SVHN_4000_NoneFunction', 'MNIST_400_SVHN_4000_StepFunctionNegative', 'MNIST_400_SVHN_4000_StepFunctionPositive', 'MNIST_400_SVHN_500_LinealFunction', 'MNIST_400_SVHN_500_NoneFunction', 'MNIST_400_SVHN_500_StepFunctionNegative', 'MNIST_400_SVHN_500_StepFunctionPositive']
    #delete_empty_folders(location_path)
    #pretraining_model(sources, location_path)

    train_with_target(targets, location_path)
    #list_folders(Path(location_path, "targets"))
      
    


    
    

if __name__ == "__main__":
    main()