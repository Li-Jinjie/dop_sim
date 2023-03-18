import numpy as np
import matplotlib.pyplot as plt

# Define the data
agent_num = np.array([100, 1e3, 3e3, 5e3, 7e3, 9e3, 1e4, 3e4, 5e4, 7e4, 9e4, 1e5])

run_time = np.array([(0.8695 + 0.8336 + 0.8539) / 3, (0.8470 + 0.8591 + 0.8518) / 3, (0.8617 + 0.8593 + 0.8768) / 3,
                     (0.9922 + 0.9990 + 0.9914) / 3, (1.2954 + 1.2875 + 1.2905) / 3, (1.5643 + 1.5950 + 1.5684) / 3,
                     (1.6915 + 1.6889 + 1.6952) / 3,
                     (4.5106 + 4.5250 + 4.5608) / 3, (4.5362 + 4.5324 + 4.5741) / 3, (11.2117 + 11.3712 + 11.2975) / 3,
                     (14.4430 + 14.3145 + 14.3145) / 3,
                     (16.0204 + 16.0319 + 16.0355) / 3])

# Create the plot
fig, ax = plt.subplots(figsize=(3.33, 2.5))

# Plot the line and add circles for each data point
ax.plot(agent_num, run_time, 'o-', markersize=2)

# Set the x-axis to log scale
ax.set_xscale('log')

# Set the labels and title
ax.set_xlabel('Number of Agents', fontsize=8)
ax.set_ylabel('Running Time Per Round (ms)', fontsize=8)
# ax.set_title('Running Time as a Function of Number of Agents', fontsize=10)

# Add a grid
ax.grid(True, which='both', linestyle='--')

# Set the font size of tick labels
ax.tick_params(axis='both', which='major', labelsize=6)

# Save the figure as a PDF file
fig.savefig('running_time_wrt_num_agent.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
