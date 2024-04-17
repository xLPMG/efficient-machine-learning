import torch
import matplotlib.pyplot as plt

## Plots the given points and colors them by the predicted labels.
#  It is assumed that a prediction larger than 0.5 corresponds to a red point.
#  All other points are black.
#  @param i_points points in R^3.
#  @param io_model model which is applied to derive the predictions.

def plot( i_points,
          io_model ):

    # switch to evaluation mode
    io_model.eval()
    with torch.no_grad(): 
        predictions = io_model(i_points)
        red_points = i_points[predictions.squeeze() > 0.5]
        black_points = i_points[predictions.squeeze() <= 0.5]
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(red_points[:, 0], red_points[:, 1], red_points[:, 2], color='red')
        ax.scatter(black_points[:, 0], black_points[:, 1], black_points[:, 2], color='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        print("(showing plot...)")
        plt.show()
