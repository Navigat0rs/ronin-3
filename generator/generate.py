import neptune.new as neptune

run = neptune.init_run(
        project="Navigator/Navigator",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYTk4NGQwYS1lMWQxLTQ3YWQtYmQ3NC1lMzBjNDVmNDI3MzAifQ==",
    )




import math
import matplotlib.pyplot as plt

start = 0.4
end = 0.0001
num_values = 1000

values = [start * math.exp(-i * math.log(start / end) / (num_values - 1)) for i in range(num_values)]
for i in values:
    run["navigator/train/batch/total_real_loss"].append(i)

# Plotting the values
plt.plot(range(num_values), values)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.show()
# # Print the generated values
#
# print(values)