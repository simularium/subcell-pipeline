# %% 
# %% [markdown]
# ### 1. Simulator Prediction
# -karthik testing idea (wip)

# %%
from subcell_analysis.cytosim.post_process_cytosim import create_dataframes_for_repeats
from subcell_analysis.compression_workflow_runner import (
    compression_metrics_workflow,
    plot_metric,
    plot_metric_list,
)
from subcell_analysis.compression_analysis import (
    COMPRESSIONMETRIC,
)
from simulariumio import ScatterPlotData

from simulariumio.cytosim import CytosimConverter
from pathlib import Path

import os
import pandas as pd
import numpy as np

import boto3
# %%
import torch 
from pathlib import Path

# %%

df_path = Path("../data/dataframes")
df = pd.read_csv(
    f"{df_path}/combined_actin_compression_metrics_all_velocities_and_repeats_subsampled_with_metrics.csv"
)
# remove first two columns from df
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace = True)

# %%
# Write a CNN to predict simulator based on xyz points of df
df
result = df[ (df['velocity'] == 15) & (df['simulator'] == 'readdy') & (df['time'] == 0.0)& (df['repeat'] == 0)]
numpy_array = result[['time', 'xpos', 'ypos', 'zpos']].to_numpy(dtype=float) 
numpy_array
# %%

tensor = torch.tensor(numpy_array, dtype=torch.float32)
print(tensor)

# %%
num_samples = 20
sequence_length = 200
num_features = 3  # XYZ

# %%
# Get unique velocities
unique_velocities = df['velocity'].unique()
unique_repeats = df['repeat'].unique()

# Initialize a dictionary to store tensors for each velocity
readdy_velocity_data = {}
# hierarchy structure is: 
# readdy_velocity_data
#   repeat_data
#        timestep
# can get the monomer points for timestep985 using readdy_velocity_data[4.7][0][985]

for velocity in unique_velocities:
    readdy_velocity_data[velocity] = {}
    for repeat in unique_repeats:
    # Filter the DataFrame for the current velocity
        result = df[(df['velocity'] == velocity) & (df['simulator'] == 'readdy')& (df['repeat'] == repeat)]
        # Group by time and convert to tensors
        if not result.empty:
            grouped_tensors = [torch.tensor(g[['xpos', 'ypos', 'zpos']].values) for _, g in result.groupby('time')]
            grouped = torch.stack(grouped_tensors)
            print(type(grouped))
            # Store the grouped data in the dictionary
            readdy_velocity_data[velocity][repeat] = grouped

# %%
# Initialize a dictionary to store tensors for each velocity

cytosim_velocity_data = {}
# hierarchy structure is: 
# readdy_velocity_data
#   repeat_data
#        timestep
# can get the monomer points for timestep985 using readdy_velocity_data[4.7][0][985]

for velocity in unique_velocities:
    cytosim_velocity_data[velocity] = {}
    for repeat in unique_repeats:
    # Filter the DataFrame for the current velocity
        result = df[(df['velocity'] == velocity) & (df['simulator'] == 'cytosim')& (df['repeat'] == repeat)]
        # Group by time and convert to tensors
        if not result.empty:
            grouped_tensors = [torch.tensor(g[['xpos', 'ypos', 'zpos']].values) for _, g in result.groupby('time')]
            grouped = torch.stack(grouped_tensors)
            print(type(grouped))
            # Store the grouped data in the dictionary
            cytosim_velocity_data[velocity][repeat] = grouped

# %%
# Initialize a list to store the concatenated tensors
simulator_tensors = []
for simulator in 
velocity_tensors = []
for velocity in unique_velocities:
    repeat_tensors = []
    for repeat in unique_repeats:
        if velocity == 15.0 and repeat == 2:
            result = cytosim_velocity_data[velocity][1]
        else:
            result = cytosim_velocity_data[velocity][repeat]
        repeat_tensors.append(result)
        print(result)
    concatenated = torch.cat(repeat_tensors, dim=0)
    velocity_tensors.append(concatenated)
# Reshape to [1000, 200, 3]
    final_shape = concatenated.reshape(-1, 200, 3)
concatenated = torch.cat(velocity_tensors, dim=0)
final_shape = concatenated.reshape(-1, 200, 3)
print(final_shape.shape)
#print("Shape of X_data:", X_data.shape)
# %%

X_data = final_shape
X_data = X_data.to(torch.float32)

# Assuming you have loaded your simulator and velocity labels
simulator_labels = np.array([0, 1])  # 0 for Simulator Cytosim, 1 for Simulator ReaDDy
velocity_labels = np.array([4.7, 15, 47, 150])  # 4 possible velocities

# Create Y_data with shape (num_samples, 2), where each row represents a sample
# Column 1: Simulator (binary encoding)
# Column 2: Velocity (integer encoding)
num_samples = len(X_data)  # Assuming X_data contains the input data
Y_data = np.zeros((num_samples, 2), dtype=np.float32)  # Initialize Y_data array

# Assign simulator and velocity labels for each sample
# For simplicity, randomly assign simulator and velocity labels here
for i in range(num_samples):
    Y_data[i, 0] = 0  # Randomly choose simulator label
    # Loop through different indices and assign velocity labels accordingly
    if i < 1000:
        Y_data[i, 1] = 4.7
    elif i < 2000:
        Y_data[i, 1] = 15
    elif i< 3000:
        Y_data[i, 1] = 47
    else:
        Y_data[i, 1] = 150

print("Y_data shape:", Y_data.shape)
print("Sample Y_data:\n", Y_data[:5, 1])  # Print first 5 samples of Y_data

# %%
from sklearn.model_selection import train_test_split

# %%
# Reshape data into [num_samples, 200, 3]
# num_samples = 200 timepoints * 5 repeats * 4 velocities

# Define your neural network architecture
import torch.nn as nn

class YourClassifier(nn.Module):
    def __init__(self):
        super(YourClassifier, self).__init__()
        # Input has 3 features (x, y, z) over 200 time steps after permute (batch_size, 3, 200)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.flatten = nn.Flatten()
        # Output size after two Conv1d layers with kernel_size=3: input_size - 2 * 2 (for each conv layer) = 200 - 4 = 196
        self.fc1 = nn.Linear(128 * 196, 64)  # Calculate the flattened size
        self.fc2 = nn.Linear(64, 1)  # Output layer for regression

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

    
X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)


# %%
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train[:, 1], dtype=torch.float32)  # Assuming velocity labels are in the second column
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val[:, 1], dtype=torch.float32) 

model = YourClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.permute(0, 2, 1))  # Permute dimensions for Conv1D layer
    loss = criterion(outputs.squeeze(), y_train_tensor)  # Assuming the output size of the last layer is 1
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor.permute(0, 2, 1))  # Permute dimensions for Conv1D layer
    val_loss = criterion(outputs.squeeze(), y_val_tensor)  # Assuming the output size of the last layer is 1
    print(f'Validation Loss: {val_loss.item()}')













# %%
