import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This file generates three datasets:
# 1. A basic sin wave dataset, essentially only global periodicity
# 2. A dataset with only local periodicity
# 3. A dataset with global periodicity and local periodicity

# Set the random seed for reproducibility
np.random.seed(0)

num_samples = 1000
time_index = pd.date_range(start="22/10/2023", periods=num_samples, freq="H")
save_folder = "dataset/constructed/"

# Make dataset directory
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Generate the basic global periodicity dataset (a sin wave)
global_periodicity = np.sin(np.linspace(0, 2*np.pi*20, num_samples))

# Save it as global_periodicity.csv
df = pd.DataFrame({"date": time_index, "Value": global_periodicity})
df.to_csv(save_folder + "global_periodicity.csv", index=False)

# Now generate the local periodicity dataset
lag_order = 3
localised_data = np.zeros(num_samples)

# Initialize the first samples from a gaussian distribution
localised_data[:lag_order] = np.random.normal(0, 1, lag_order)

# Generate the remaining points using a weighted sum of the previous data points + a random value
for i in range(lag_order, num_samples):
    localised_data[i] = 0.7 * localised_data[i-1] + 0.4 * localised_data[i-2] - 0.4 * localised_data[i-3] + np.random.normal(0, 0.5)

# smooth the data with a moving average
window = 10
smoothed_data = np.convolve(localised_data, np.ones(window) / window, mode='same')

# save the localised data as local_periodicity.csv
df = pd.DataFrame({"date": time_index, "Value": localised_data})
df.to_csv(save_folder + "local_periodicity.csv", index=False)

# combine both datasets to create the dataset with local and global feature patterns
global_local_periodicity = global_periodicity + localised_data

# Save it as global_local_periodicity.csv
df = pd.DataFrame({"date": time_index, "Value": global_local_periodicity})
df.to_csv(save_folder + "global_local_periodicity.csv", index=False)

# Now finally, plot all three datasets separately
plt.figure(figsize=(20, 10))
plt.plot(time_index, global_periodicity)
plt.title("Global Periodicity Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig(save_folder + "global_periodicity.png")
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(time_index, localised_data)
plt.title("Local Periodicity Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig(save_folder + "local_periodicity.png")
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(time_index, global_local_periodicity)
plt.plot(time_index, global_periodicity, linestyle="--", label="Underlying Global Periodicity")
plt.title("Global and Local Periodicity Dataset")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig(save_folder + "global_local_periodicity.png")
plt.show()
