import torch

def low_pass_filter(input_tensor, alpha):
    # batch_size, sequence_length, feature_dim = input_tensor.shape
    batch_size, feature_dim = input_tensor.shape

    # Initialize filtered input
    filtered_input = torch.zeros_like(input_tensor, device='cuda:0')

    # Apply low-pass filtering
    # filtered_input[:, 0] = input_tensor[:, 0]
    filtered_input[0] = input_tensor[0]

    # for t in range(1, sequence_length):
    #     filtered_input[:, t] = alpha * input_tensor[:, t] + (1 - alpha) * filtered_input[:, t-1]
    for i in range (1,len(filtered_input)):
        filtered_input[i] = alpha * input_tensor[i] + (1 - alpha) * filtered_input[i - 1]

    return filtered_input