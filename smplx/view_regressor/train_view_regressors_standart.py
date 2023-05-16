import os
import argparse
import time
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

from model import RegressionModel, SphericalDistanceLoss
from smplx.view_regressor.data_processing import load_data_from_coco_file, process_keypoints, c2s, s2c
from visualizations import plot_training_data

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument('--views-filename', type=str, default="views.json",
                        help='Filename of the views file')
    parser.add_argument('--coco-filename', type=str, default="person_keypoints_val2017.json",
                        help='Filename of the coco annotations file')
    parser.add_argument('--workdir', type=str, default="view_regressor_workdir",
                        help='Workdir where to save the model and the logs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--spherical-output', action="store_true", default=False,
                        help='If True, will train the regressor on spherical coordinates ignoring the radius')
    parser.add_argument('--loss', type=str, default="MSE",
                        help='Loss function. Known values: MSE, L1, Spherical')
    parser.add_argument('--load', action="store_true", default=False,
                        help='If True, will load the model from the checkpoint file')
    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Will force CPU computation')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Will print loss to the console')
    
    return parser.parse_args()

def main(args):

    os.makedirs(args.workdir, exist_ok=True)

    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)
    
    # Load the data
    keypoints, bboxes_xywh, image_ids, positions = load_data_from_coco_file(coco_filepath, views_filepath)
    keypoints = process_keypoints(keypoints, bboxes_xywh)    
    
    if args.spherical_output:
        positions = c2s(positions)

    # If CUDA available, use it
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    input_size = keypoints.shape[1]
    output_size = positions.shape[1]

    # Create a DataLoader
    dataset = TensorDataset(
        torch.from_numpy(keypoints).float(),
        torch.from_numpy(positions).float(),
    )

    # Split the data to the training and testing sets
    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_size if args.batch_size <= 0 else args.batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    # Define the model, loss function, and optimizer
    model = RegressionModel(input_size=input_size, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    if args.loss.upper() == "MSE":
        criterion = nn.MSELoss()
    elif args.loss.upper() == "L1":
        criterion = nn.L1Loss()
    elif args.loss.upper() == "SPHERICAL":
        if not args.spherical_output:
            criterion = nn.L1Loss()
            warnings.warn("Spherical loss function used with cartesian output. Regressing to the L1 loss")
        else:
            criterion = SphericalDistanceLoss()
    else:
        raise ValueError("Unknown loss function: {}".format(args.loss))

    # Print the number of parameters
    print('Number of parameters: {}'.format(model.count_parameters()))

    # Training loop
    writer = SummaryWriter(
        log_dir=os.path.join(args.workdir, "tensorboar_logs"),
    )
    start_time = time.time()
    for epoch in tqdm(range(args.epochs)):
        for batch_x, batch_y in train_dataloader:
            
            # Forward pass
            y_pred = model(batch_x.to(device))
            loss = criterion(y_pred, batch_y.to(device))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss
            writer.add_scalar('Loss/train', loss.item(), epoch)

        # Test the model
        if epoch % args.test_interval == 0:
            with torch.no_grad():
                for batch_x, batch_y in test_dataloader:
                    y_pred = model(batch_x.to(device))
                    loss = criterion(y_pred, batch_y.to(device))
                    writer.add_scalar('Loss/test', loss.item(), epoch)
        
        # Print progress
        if args.verbose and (epoch+1) % 10 == 0:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (epoch+1) * (args.epochs - epoch - 1)
            print('Epoch [{:5d}/100]\tLoss: {:7.4f}\tElapsed: {:5.2f} s\tRemaining: {:5.2f} s'.format(epoch+1, loss.item(), elapsed_time, remaining_time))

    # Test the model
    with torch.no_grad():
        for batch_x, batch_y in test_dataloader:
            y_pred = model(batch_x.to(device))
            loss = criterion(y_pred, batch_y.to(device))
            writer.add_scalar('Loss/test', loss.item(), epoch)
            print('Test loss: {:.4f}'.format(loss.item()))

    # Save the model
    model_filename = os.path.join(args.workdir, "view_regressor.pth")
    torch.save(model.cpu().state_dict(), model_filename)
    print("Model saved to {}".format(model_filename))


if __name__ == '__main__':
    args = parse_args()
    main(args)