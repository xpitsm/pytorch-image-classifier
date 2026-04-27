# Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)

import sys
from dataset import CLADataset
import torch
from skimage import io

from sklearn.utils import shuffle
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import csv
import pandas as pd  # To compute acc

from network import ResNet, BasicBlock


def compute_accuracy():
    # Load the reference predictions and model predictions CSV files
    reference_df = pd.read_csv("reference/reference_predictions.csv")
    predictions_df = pd.read_csv("output_predictions/predictions.csv")

    # Merge the dataframes based on the 'filename' column
    merged_df = pd.merge(reference_df, predictions_df, on='filename')

    # Compute accuracy
    accuracy = (merged_df['class_id_x'] == merged_df['class_id_y']).mean()

    print(f"Accuracy: {accuracy}")


def create_testing_data(test_path, test_images, test_paths):
    """
    Create testing data by loading images from the specified directory.

    Parameters:
        - test_path (str): Path to the directory containing test images. Expects all imgs in one folder.
        - test_images (list): List to store test images.
        - test_paths (list): List to store paths of test images.

    Returns:
        - None
    """

    class_names = ['bus', 'car', 'light', 'sign', 'truck', 'vegetation']

    for i, class_name in enumerate(class_names):  # Go through particular directories
        class_dir = os.path.join(test_path, class_name)
        class_files = os.listdir(class_dir)
        for file in class_files:  # Go through imgs
            img_path = os.path.join(class_dir, file)
            test_paths.append(file)

            img = io.imread(img_path)
            test_images.append(img)  # Store img as numpy array


# sample function for performing inference for a whole dataset that is sent
def infer_all(model, test_dl, results, dev):
    """
    Perform inference for all samples in the test dataloader.

    Parameters:
        - model (torch.nn.Module): The neural network model.
        - test_dl (torch.utils.data.DataLoader): DataLoader for test data.
        - results (list): List to store inference results.
        - dev: The device (CPU or GPU) used to perform inference.

    Returns:
        None
    """

    # Do not calculate the gradients
    with torch.no_grad():
        for sample in tqdm(test_dl, total=len(test_dl)):
            sample = sample.to(dev)
            pred = model(sample)
            res = pred.argmax(axis=1)

            results += list(res.detach().cpu().numpy())
    return


def write_predictions(test_paths, results):
    """
    Write predictions to a CSV file.

    Parameters:
        test_paths (list): List of test image paths.
        results (list): List of predicted class indices.

    Returns:
        None
    """

    # Create a list of tuples with filename (img name) and predicted class
    output_data = list(zip(test_paths, results))

    # Create output directory if it doesn't exist
    output_dir = 'output_predictions'
    os.makedirs(output_dir, exist_ok=True)

    # Write data to CSV file
    output_file = os.path.join(output_dir, 'predictions.csv')
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'class_id'])  # Write header
        writer.writerows(output_data)  # Write data

    print('\n Predictions saved to ``' + output_file + '``')


# declaration for this function should not be changed
def inference(dataset_path, model_path, n_samples=0):
    """
    inference(dataset_path, model_path='model.pt') performs inference on the given dataset;
    if n_samples is not passed or <= 0, the predictions are performed for all data samples at the dataset_path
    if  n_samples=N, where N is an int positive number, then only N first predictions are performed
    saves:
    - predictions to 'output_predictions' folder

    Parameters:
    - dataset_path (string): path to a dataset
    - model_path (string): path to a model
    - n_samples (int): optional parameter, number of predictions to perform

    Returns:
    - None
    """
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    # Creating testing data
    test_path = dataset_path
    test_paths = []
    test_images = []
    create_testing_data(test_path, test_images, test_paths)

    # Shuffle, not to have sequentially read imgs separated in classes
    test_images_b = test_images.copy()
    test_paths_b = test_paths.copy()
    test_images_b, test_paths_b = shuffle(test_images_b, test_paths_b, random_state=42)

    # Set batch size and img size
    batch_size = 1
    img_size = 256

    # Define transforms for test data
    transforms_test = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, always_apply=True, border_mode=0),
        ToTensorV2(),
    ]
    )

    # Create test dataset and test dataloader
    test_ds = CLADataset(transforms_test, test_images_b)
    test_dl = torch.utils.data.DataLoader(
        test_ds,
        shuffle=False,
        batch_size=batch_size)

    # Loading the model
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Set to eval mode
    model.eval()

    print('Predicting on test set')
    results = []
    # If n_samples <= 0 -> perform predictions all data samples at the dataset_path
    if n_samples <= 0:
        infer_all(model, test_dl, results, device)
    else:
        # Perform predictions only for the first n_samples images
        with torch.no_grad():
            which_sample = 1
            for sample in tqdm(test_dl, total=len(test_dl)):
                sample = sample.to(device)
                pred = model(sample)
                res = pred.argmax(axis=1)
                results += list(res.detach().cpu().numpy())

                if which_sample == n_samples:
                    test_paths_b = test_paths_b[:n_samples]
                    break

                which_sample += 1

    # Write predictions to predictions.csv
    write_predictions(test_paths_b, results)

    # Compute acc
    compute_accuracy()
    return


def get_arguments():
    if len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        model_path = sys.argv[2]
        number_of_samples = 0
    elif len(sys.argv) == 4:
        try:
            dataset_path = sys.argv[1]
            model_path = sys.argv[2]
            number_of_samples = int(sys.argv[3])
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print("Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)")
        sys.exit(1)

    return dataset_path, model_path, number_of_samples


if __name__ == "__main__":
    path_2_dataset, path_2_model, n_samples_2_predict = get_arguments()
    inference(path_2_dataset, path_2_model, n_samples_2_predict)
