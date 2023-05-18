import torch

def kalman_filter(input_tensor):
    batch_size, sequence_length, feature_dim = input_tensor.shape

    # Define Kalman filter parameters
    state_dim = feature_dim // 2  # Assuming equal dimensions for linear acceleration and angular velocity
    transition_matrix = torch.eye(state_dim, device='cuda:0')
    transition_covariance = torch.eye(state_dim, device='cuda:0')
    observation_matrix = torch.eye(feature_dim, state_dim, device='cuda:0')
    observation_covariance = torch.eye(feature_dim, device='cuda:0')
    initial_state_mean = torch.zeros(state_dim, device='cuda:0')
    initial_state_covariance = torch.eye(state_dim, device='cuda:0')

    # Initialize filtered input
    filtered_input = torch.zeros_like(input_tensor, device='cuda:0')

    # Perform Kalman filtering for each batch and time step
    for i in range(batch_size):
        current_state_mean = initial_state_mean.unsqueeze(0).expand(sequence_length, -1).clone()
        current_state_covariance = initial_state_covariance.unsqueeze(0).expand(sequence_length, -1, -1).clone()

        for t in range(sequence_length):
            observation = input_tensor[i, t]

            # Predict step
            predicted_state_mean = torch.matmul(transition_matrix, current_state_mean[t].t()).t()
            predicted_state_covariance = torch.matmul(transition_matrix, torch.matmul(current_state_covariance[t], transition_matrix.t())) + transition_covariance

            # Update step
            innovation = observation - torch.matmul(observation_matrix, predicted_state_mean.t()).t()
            innovation_covariance = torch.matmul(observation_matrix, torch.matmul(predicted_state_covariance, observation_matrix.t())) + observation_covariance
            kalman_gain = torch.matmul(predicted_state_covariance, torch.matmul(observation_matrix.t(), torch.inverse(innovation_covariance)))
            current_state_mean[t] = predicted_state_mean + torch.matmul(kalman_gain, innovation.t()).t()
            current_state_covariance[t] = predicted_state_covariance - torch.matmul(kalman_gain, torch.matmul(observation_matrix, predicted_state_covariance))

            filtered_input[i, t] = torch.matmul(observation_matrix, current_state_mean[t].t()).t()

    return filtered_input

import torch
from torch import nn

import torch
from torch import nn

class KalmanFilter(nn.Module):
    def __init__(self, input_dim, device):
        super(KalmanFilter, self).__init__()
        self.input_dim = input_dim
        self.device = device

        # Kalman filter parameters
        self.transition_matrix = nn.Parameter(torch.eye(input_dim, input_dim))
        self.transition_covariance = nn.Parameter(torch.eye(input_dim, input_dim))
        self.observation_matrix = nn.Parameter(torch.eye(input_dim, input_dim))
        self.observation_covariance = nn.Parameter(torch.eye(input_dim, input_dim))

        # Initialize state estimate and error covariance
        self.state_estimate = nn.Parameter(torch.zeros(input_dim, 1))
        self.error_covariance = nn.Parameter(torch.eye(input_dim, input_dim))

    def forward(self, input_tensor):
        # Move the model parameters to the desired device
        self.to(self.device)

        # Convert input tensor to device
        input_tensor = input_tensor.to(self.device)

        batch_size = input_tensor.size(0)
        filtered_output = torch.empty_like(input_tensor)

        # Apply the Kalman filter to each sample in the batch
        for i in range(batch_size):
            # Initialize the state estimate and error covariance
            state_estimate = self.state_estimate.clone()
            error_covariance = self.error_covariance.clone()

            for j in range(self.input_dim):
                # Update the state estimate and error covariance based on the transition model
                state_estimate = torch.matmul(self.transition_matrix, state_estimate)
                error_covariance = torch.matmul(
                    torch.matmul(self.transition_matrix, error_covariance),
                    torch.transpose(self.transition_matrix, 0, 1)
                ) + self.transition_covariance

                # Calculate the Kalman gain
                kalman_gain = torch.matmul(
                    torch.matmul(error_covariance, torch.transpose(self.observation_matrix, 0, 1)),
                    torch.inverse(
                        torch.matmul(
                            torch.matmul(self.observation_matrix, error_covariance),
                            torch.transpose(self.observation_matrix, 0, 1)
                        ) + self.observation_covariance
                    )
                )

                # Update the state estimate based on the observation
                observation = input_tensor[i][j].unsqueeze(0)
                state_estimate += torch.matmul(kalman_gain, observation - torch.matmul(self.observation_matrix, state_estimate))

                # Update the error covariance
                error_covariance = torch.matmul(
                    torch.eye(self.input_dim).to(self.device) - torch.matmul(kalman_gain, self.observation_matrix),
                    error_covariance
                )

                # Store the filtered output
                filtered_output[i, j] = state_estimate.squeeze(dim=-1)

        return filtered_output



# Example usage

# Example usage
# input_tensor = torch.randn(128, 2)  # Input tensor of shape (128, 2)
# device = torch.device("cuda:0")  # Device to use (GPU)
#
# # Create an instance of the Kalman filter
# kalman_filter = KalmanFilter(input_dim=2, device=device)
#
# # Apply the Kalman filter to the input tensor
# filtered_output = kalman_filter(input_tensor)
#
# # Print the shape of the filtered output tensor
# print(filtered_output.shape)

