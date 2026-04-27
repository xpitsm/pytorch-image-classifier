# Usage: python training.py <path_2_dataset>

import sys
import matplotlib.pyplot as plt
import csv

import torch
from torchview import draw_graph
from network import ResNet, BasicBlock
from dataset import CLADataset
from torch.utils.data import DataLoader

import os
from skimage import io  # reading img
from sklearn.utils import shuffle  # shuffle before split
from sklearn.model_selection import train_test_split  # split data
import albumentations as A  # for transformations
from albumentations.pytorch import ToTensorV2  # obtain tensor from numpy array

from tqdm import tqdm  # tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau  # scheduler
import numpy as np  # to work with numpy

################################################################################
# CREATE TRAIN TEST DATASET                                                    #
################################################################################

# This flag is now set to False, for splitting the whole dataset to train & test -> set it to True
TRAIN_TEST_FLAG = False

# This flag is now set to False, but when fine-tuning I tried to train the model both with class weights
# (to address class imbalance) and without them
USE_CLASS_WEIGHTS_FLAG = False


def load_data(data_dir, class_names):
    """
    Load data from specified directory.

    Parameters:
        - data_dir (str): Path to the directory containing the data.
        - class_names (list): List of class names.

    Returns:
        - tuple: A tuple containing file paths and corresponding labels.
    """

    file_paths = []
    labels = []

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        class_files = os.listdir(class_dir)
        file_paths += [os.path.join(class_dir, file) for file in class_files]
        labels += [class_name] * len(class_files)  # Assign labels based on class

    # Shuffle data
    file_paths, labels = shuffle(file_paths, labels, random_state=42)
    return file_paths, labels


def move_images(class_names, dest_dir, X, y):
    """
    Move images to the specified destination directory based on their labels.

    Parameters:
        - class_names (list): List of class names.
        - dest_dir (str): Path to the destination directory where we want to move the images.
        - X (list): List of file paths of images.
        - y (list): List of labels corresponding to images.
    """
    # Create class directories
    for class_name in class_names:
        path = dest_dir + '/' + class_name
        os.makedirs(path, exist_ok=True)

    # Move images to the destination directory to corresponding directories based on class
    for src_path, label in zip(X, y):
        class_name = label  # Class name is directly taken from y_test
        dst_path = os.path.join(dest_dir, class_name, os.path.basename(src_path))  # Destination path
        os.replace(src_path, dst_path)  # Move the image to the test directory


def create_reference_file(X_test, y_test):
    """
    Create reference CSV file for evaluation (ground truth).

    Parameters:
        - X_test (list): List of file paths of test images.
        - y_test (list): List of labels corresponding to test images.

    """
    class_dict = {
        'bus': 0,
        'car': 1,
        'light': 2,
        'sign': 3,
        'truck': 4,
        'vegetation': 5}

    # Create reference csv (that can be used in evaluation.py script)
    reference_classes = [class_dict[ref] for ref in y_test]
    X_test_base = [os.path.basename(file_path) for file_path in X_test]

    # Create a list of tuples with filename and predicted class
    reference_data = list(zip(X_test_base, reference_classes))

    # Write data to CSV file
    reference_file = "reference_predictions.csv"
    print('File with reference predictions saved as ``' + reference_file + '``')
    with open(reference_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'class_id'])  # Write header
        writer.writerows(reference_data)  # Write data


def create_train_test_data(data_dir, class_names):
    """
    Create train and test directories, split data, and move images accordingly.

    Parameters:
        - data_dir (str): Path to the directory containing the data.
        - class_names (list): List of class names.
    """

    train_dir = "train"
    test_dir = "test"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    file_paths, labels = load_data(data_dir, class_names)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2,
                                                        random_state=42)  # tried with straitified=labels, obtained pretty much same results

    # Move test imgs, create ground truth and move train imgs
    move_images(class_names, test_dir, X_test, y_test)
    create_reference_file(X_test, y_test)
    move_images(class_names, train_dir, X_train, y_train)


################################################################################
# TRAINING                                                                     #
################################################################################

def draw_network_architecture(network, input_sample):
    """
    Save visualization of model architecture to model_architecture.png.

    Parameters:
        - network (torch.nn.Module): The neural network model.
        - input_sample (torch.Tensor): Sample input tensor.

    Returns:
        - None
    """

    model_graph = draw_graph(network, input_sample, graph_dir='LR')
    model_graph.visual_graph.attr(size='500,500')  # Adjust size
    model_graph.visual_graph.render('model_architecture', format='png', cleanup=True)
    print('Model architecture saved as ``model_architecture.png``')


def plot_learning_curves(train_losses, validation_losses):
    """
    Plot learning curves showing train and validation losses.

    Parameters:
        - train_losses (list): List of training losses.
        - validation_losses (list): List of validation losses.

    Returns:
        - None
    """

    plt.figure(figsize=(10, 5))
    plt.title("Train and Evaluation Losses During Training")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("learning_curves.png")


def create_training_data(train_path, images, labels):
    """
    Create training data by loading images and labels.

    Parameters:
        - train_path (str): Path to the training data directory.
        - images (list): List to store images.
        - labels (list): List to store labels.

    Returns:
        - None
    """

    class_names = ['bus', 'car', 'light', 'sign', 'truck', 'vegetation']

    for i, class_name in enumerate(class_names):  # Go through particular directories
        class_dir = os.path.join(train_path, class_name)
        class_files = os.listdir(class_dir)
        for file in class_files:  # Go through imgs
            img_path = os.path.join(class_dir, file)
            img = io.imread(img_path)
            images.append(img)  # Store img as numpy array
            labels.append(i)  # Store label


def calculate_class_weights(y_train):
    """
    Calculate class weights based on the distribution of class labels in the training data.

    Parameters:
        - y_train (list): List of class labels.

    Returns:
        - dict: A dictionary containing class weights for each unique class label.
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    class_weights = {}

    for class_label, class_count in zip(unique_classes, class_counts):
        class_weight = total_samples / (len(unique_classes) * class_count)
        class_weights[class_label] = class_weight

    return class_weights


"""
Training loop, functionality related to fit()
"""


def train(model, train_dl, loss_func, dev, opt):
    """
    Train the model.

    Parameters:
        model (torch.nn.Module): The neural network model.
        train_dl (torch.utils.data.DataLoader): DataLoader for training data.
        loss_func: The loss function.
        dev: The device (CPU or GPU) used to perform training.
        opt: The optimizer.

    Returns:
        float: The average loss over the training set.
    """

    model.train()
    total_loss, total_size = 0, 0
    for b_idx, (xb, yb) in tqdm(enumerate(train_dl), total=len(train_dl), leave=False):
        batch_loss, batch_size = loss_batch(model, loss_func, xb, yb, dev, opt)
        total_loss += batch_loss * batch_size
        total_size += batch_size
    return total_loss / total_size


def validate(model, valid_dl, loss_func, dev):
    """
    Validate the model.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - valid_dl (torch.utils.data.DataLoader): DataLoader for validation data.
        - loss_func: The loss function.
        - dev: The device (CPU or GPU) used to perform validation.

    Returns:
        tuple: A tuple containing validation loss and validation accuracy.
    """

    model.eval()
    total_loss, total_correct, total_size = 0, 0, 0
    with torch.no_grad():
        for xb, yb in valid_dl:
            xb, yb = xb.to(dev), yb.to(dev)
            output = model(xb)
            loss = loss_func(output, yb)
            total_loss += loss.item() * len(xb)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == yb).sum().item()
            total_size += len(xb)
    val_loss = total_loss / total_size
    val_acc = total_correct / total_size
    return val_loss, val_acc


def loss_batch(model, loss_func, xb, yb, dev, opt=None):
    """
    Calculate loss for a batch.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - loss_func: The loss function.
        - xb: The input batch.
        - yb: The target batch.
        - dev: The device (CPU or GPU) used to perform computations.
        - opt: The optimizer.

    Returns:
        - tuple: A tuple containing the batch loss and batch size.
    """

    xb, yb = xb.to(dev), yb.to(dev)
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, dev, scheduler=None):
    """
    Train the model for a specified number of epochs.

    Parameters:
        - epochs (int): Number of epochs to train for.
        - model (torch.nn.Module): The neural network model.
        - loss_func: The loss function.
        - opt: The optimizer.
        - train_dl (torch.utils.data.DataLoader): DataLoader for training data.
        - valid_dl (torch.utils.data.DataLoader): DataLoader for validation data.
        - dev: The device (CPU or GPU) used to perform training.
        - scheduler: Learning rate scheduler. Default is None.

    Returns:
        - tuple: A tuple containing lists of training and validation losses.
    """

    train_losses, val_losses = [], []
    best_val_loss = float('inf')  # Set the initial best validation loss to infinity

    for epoch in tqdm(range(epochs)):

        # Training step
        train_loss = train(model, train_dl, loss_func, dev, opt)
        train_losses.append(train_loss)

        # Evaluation step
        val_loss, val_acc = validate(model, valid_dl, loss_func, dev)
        val_losses.append(val_loss)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        # Save model based on best_val_loss
        if val_loss < best_val_loss:  # Check if validation loss has improved
            print("\ncurr_best_val_loss: " + str(best_val_loss) + " " + "substituted for: " + str(val_loss))
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pt')

        # Print metrics
        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss : .05f}, Validation Loss: {val_loss:.05f}, Validation Accuracy: {val_acc:.05f}')

    return train_losses, val_losses


# declaration for this function should not be changed
def training(dataset_path):
    """
    training(dataset_path) performs training on the given dataset;
    saves:
    - model.pt (trained model)
    - learning_curves.png (learning curves generated during training)
    - model_architecture.png (a scheme of model's architecture)

    Parameters:
    - dataset_path (string): path to a dataset

    Returns:
    - None
    """

    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    if TRAIN_TEST_FLAG:
        print("``TRAIN_TEST_FLAG`` set to True, creating train test data")
        class_names = ['bus', 'car', 'light', 'sign', 'truck', 'vegetation']
        data_dir = 'data_cla_public'
        create_train_test_data(data_dir, class_names)
    else:
        print("``TRAIN_TEST_FLAG`` set to False, proceeding in training...\n")

    # Create train data
    train_path = dataset_path
    images = []
    labels = []
    create_training_data(train_path, images, labels)

    # Shuffle train data
    # Seed 42 for reproducibility
    images, labels = shuffle(images, labels, random_state=42)

    # Split train data to train and valid
    X_train, X_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.1, random_state=42)

    # Set batch size and img size
    batch_size = 32
    img_size = 256

    # Define transforms for train data
    transforms_train = A.Compose([
        # Augmentation
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.2),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
        A.LongestMaxSize(max_size=img_size),  # Longer side of img will have img_size; preserves aspect ratio
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
        # Pad if needed to have img of size img_size x img_size

        ToTensorV2(),  # Transform to tensor
    ])

    # Define transforms for valid data
    transforms_valid = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # Normalize
        A.LongestMaxSize(max_size=img_size),  # Longer side of img will have img_size; preserves aspect ratio
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
        # Pad if needed to have img of size img_size x img_size
        ToTensorV2(),  # Transform to tensor
    ]
    )

    # Create train and valid datasets
    train_ds = CLADataset(transforms=transforms_train, images=X_train, labels=y_train)
    valid_ds = CLADataset(transforms=transforms_valid, images=X_valid, labels=y_valid)

    # Create train and valid dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)

    valid_dl = torch.utils.data.DataLoader(valid_ds,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=1)

    # Create model and draw its architecture
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=6).to(device)

    input_sample = torch.zeros((1, 3, 256, 256))
    draw_network_architecture(model, input_sample)

    # Define optimizer, learning rate and scheduler
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # Define loss function
    # If the following flag is true, compute class weights and use them in loss fc
    if USE_CLASS_WEIGHTS_FLAG:
        print('``CLASS_WEIGHTS_FLAG`` set to True, computing class weights and using them in training...')
        # Compute weights
        class_weights = calculate_class_weights(y_train)

        # Convert the class weights to a PyTorch tensor
        weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
        weights_tensor = weights_tensor.to(device)
        print(weights_tensor)

        loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    # Define number of epochs
    epochs = 30

    # Train the network for thirty epochs
    train_losses, valid_losses = fit(epochs, model, loss_fn, optimizer, train_dl, valid_dl, device, scheduler)

    # Plot the losses
    plot_learning_curves(train_losses, valid_losses)

    return


def get_arguments():
    if len(sys.argv) != 2:
        print("Usage: python training.py <path_2_dataset> ")
        sys.exit(1)

    try:
        path = sys.argv[1]
    except Exception as e:
        print(e)
        sys.exit(1)
    return path


if __name__ == "__main__":
    path_2_dataset = get_arguments()
    training(path_2_dataset)
