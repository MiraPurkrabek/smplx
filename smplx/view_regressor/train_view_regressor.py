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
from smplx.view_regressor.data_processing import load_data_from_coco_file, process_keypoints, c2s, s2c, angular_distance
from visualizations import plot_training_data, plot_heatmap

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
                        help='If True, will train the regressor on spherical coordinates with the radius')
    parser.add_argument('--flat-output', action="store_true", default=False,
                        help='If True, will train the regressor on spherical coordinates ignoring the radius')
    parser.add_argument('--loss', type=str, default="MSE",
                        help='Loss function. Known values: MSE, L1, Spherical')
    parser.add_argument('--distance', type=str, default="Euclidean",
                        help='Distance function. Known values: Euclidean, Spherical. If Spherical adn 3d output, will ignore the radius.')
    parser.add_argument('--load', action="store_true", default=False,
                        help='If True, will load the model from the checkpoint file')
    parser.add_argument('--cpu', action="store_true", default=False,
                        help='Will force CPU computation')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='Will print loss to the console')
    
    return parser.parse_args()


def test_model(args, model, dataloader, device, criterion, epoch, writer):
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x.to(device))
            
            # Log the loss
            loss = criterion(y_pred, batch_y.to(device))
            writer.add_scalar('Loss/test', loss.item(), epoch)

            # Log the distance
            if args.distance.upper() == "EUCLIDEAN":
                test_distance = np.linalg.norm(y_pred.cpu().numpy() - batch_y.cpu().numpy(), axis=1)
            elif args.distance.upper() == "SPHERICAL":
                test_distance = angular_distance(y_pred.cpu().numpy(), batch_y.cpu().numpy())
            writer.add_histogram(
                'Test distance/test',
                test_distance,
                global_step = epoch,
            )

            # Log the histogram of radius
            if args.spherical_output:
                test_radius = y_pred[:, 0].cpu().numpy()
            else:
                test_radius = np.linalg.norm(y_pred.cpu().numpy(), axis=1)
            writer.add_histogram(
                'Test radius/test',
                test_radius,
                global_step = epoch,
            )

            # Log the PDF (probability density function)
            test_pdf = plot_heatmap(y_pred.cpu().numpy(), args.spherical_output, return_img=True)
            test_pdf = np.array(test_pdf).astype(np.uint8).transpose(2, 0, 1)
            writer.add_image(
                "Test PDF/test",
                test_pdf,
                global_step = epoch,
            )


def main(args):

    os.makedirs(args.workdir, exist_ok=True)

    views_filepath = os.path.join(args.folder, args.views_filename)
    coco_filepath = os.path.join(args.folder, args.coco_filename)
    
    # Load the data
    keypoints, bboxes_xywh, image_ids, positions = load_data_from_coco_file(coco_filepath, views_filepath)
    keypoints = process_keypoints(keypoints, bboxes_xywh)    
    
    if args.spherical_output:
        positions = c2s(positions)
    elif args.flat_output:
        positions = c2s(positions)
        positions = positions[:, 1:]

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
    
    if not args.distance.upper() in ["EUCLIDEAN", "SPHERICAL"]:
        raise ValueError("Unknown distance function: {}".format(args.distance))
    
    # Print the number of parameters
    print('Number of parameters: {}'.format(model.count_parameters()))

    # Training loop
    writer = SummaryWriter(
        log_dir=os.path.join(args.workdir, "tensorboard_logs"),
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
            test_model(
                args,
                model,
                test_dataloader,
                device,
                criterion,
                epoch,
                writer,
            )
        
        # Print progress
        if args.verbose and (epoch+1) % 10 == 0:
            elapsed_time = time.time() - start_time
            remaining_time = elapsed_time / (epoch+1) * (args.epochs - epoch - 1)
            print('Epoch [{:5d}/100]\tLoss: {:7.4f}\tElapsed: {:5.2f} s\tRemaining: {:5.2f} s'.format(epoch+1, loss.item(), elapsed_time, remaining_time))


    # Test the model
    test_model(
        args,
        model,
        test_dataloader,
        device,
        criterion,
        epoch,
        writer,
    )
            

    # Save the model
    model_filename = os.path.join(args.workdir, "view_regressor.pth")
    torch.save(model.cpu().state_dict(), model_filename)
    print("Model saved to {}".format(model_filename))


if __name__ == '__main__':
    args = parse_args()
    main(args)