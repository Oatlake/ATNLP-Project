import numpy as np
import matplotlib.pyplot as plt

# Dataset sizes and corresponding mean token accuracies & standard deviations
sizes = ["p1", "p2", "p4", "p8", "p16", "p32", "p64"]
mean_acc = [0.0000, 0.0000, 0.0000, 0.4703, 0.4989, 0.5000, 0.5367]
std_dev  = [0.0000, 0.0000, 0.0000, 0.0193, 0.0022, 0.0083, 0.0014]

# Convert accuracies to percentages
mean_acc_percent = [m * 100 for m in mean_acc]
std_dev_percent  = [s * 100 for s in std_dev]

x = np.arange(len(sizes))
plt.figure(figsize=(7, 4))  
width = 0.6  
plt.bar(
    x, 
    mean_acc_percent, 
    width=width,
    yerr=std_dev_percent,
    align='center',
    alpha=0.7,
    ecolor='black',
    capsize=5,
    color='skyblue',
    edgecolor='black'
)

plt.xlabel('Percent of commands used for training')
plt.ylabel('Mean Token Accuracy (%)')
plt.xticks(x, sizes)
plt.ylim([0, 100])
plt.tight_layout()
plt.show()





# Dataset labels
sizes = ["p8", "p16", "p32", "p64"]

# Mean teacher-forcing accuracies and standard deviations
mean_acc =  [0.8887,  0.8908,  0.9945,  0.8821]
mean_std =  [0.0083,  0.0047,  0.0005,  0.0091]

# Mean greedy accuracies and standard deviations
greedy_acc = [0.6825, 0.6912, 0.9868, 0.6689]
greedy_std = [0.0260, 0.0200, 0.0033, 0.0247]

# Convert to percentages
mean_acc_percent   = [m * 100 for m in mean_acc]
mean_std_percent   = [s * 100 for s in mean_std]
greedy_acc_percent = [g * 100 for g in greedy_acc]
greedy_std_percent = [s * 100 for s in greedy_std]

x = np.arange(len(sizes))  
width = 0.35  
plt.figure(figsize=(7, 4))
plt.bar(
    x - width/2, 
    mean_acc_percent,
    width=width,
    yerr=mean_std_percent,
    ecolor='black',
    capsize=5,
    color='skyblue',
    edgecolor='black',
    alpha=0.7,
    label='Mean Accuracy'
)

plt.bar(
    x + width/2, 
    greedy_acc_percent,
    width=width,
    yerr=greedy_std_percent,
    ecolor='black',
    capsize=5,
    color='lightgreen',
    edgecolor='black',
    alpha=0.7,
    label='Greedy Accuracy'
)

plt.xticks(x, sizes)               
plt.xlabel('Percent of commands used for training (T5-small)')
plt.ylabel('Mean Token Accuracy (%)')
plt.ylim([0, 100])                 
plt.title('Experiment 1: Mean vs. Greedy Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


# Dataset labels
sizes = [
    "1", 
    "2", 
    "4", 
    "8", 
    "16", 
    "32"
]

# Mean teacher-forcing accuracies and standard deviations
mean_acc =  [0.7450, 0.7905, 0.8506, 0.8469, 0.9077, 0.9485]
mean_std =  [0.0127, 0.0095, 0.0064, 0.0201, 0.0143, 0.0084]

# Mean greedy accuracies and standard deviations
greedy_acc = [0.6556, 0.6793, 0.7338, 0.7581, 0.8112, 0.9014]
greedy_std = [0.0335, 0.0209, 0.0282, 0.0203, 0.0192, 0.0137]

# Convert to percentages
mean_acc_percent   = [m * 100 for m in mean_acc]
mean_std_percent   = [s * 100 for s in mean_std]
greedy_acc_percent = [g * 100 for g in greedy_acc]
greedy_std_percent = [s * 100 for s in greedy_std]

x = np.arange(len(sizes))  
width = 0.35  
plt.figure(figsize=(8, 4))
plt.bar(
    x - width/2, 
    mean_acc_percent,
    width=width,
    yerr=mean_std_percent,
    ecolor='black',
    capsize=5,
    color='skyblue',
    edgecolor='black',
    alpha=0.7,
    label='Mean Accuracy'
)

plt.bar(
    x + width/2, 
    greedy_acc_percent,
    width=width,
    yerr=greedy_std_percent,
    ecolor='black',
    capsize=5,
    color='lightgreen',
    edgecolor='black',
    alpha=0.7,
    label='Mean Greedy Accuracy'
)

plt.xticks(x, sizes)              
plt.xlabel('Number of composed commands')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 100])                 
plt.title('Experiment 3: Mean vs. Greedy Accuracy')
plt.legend()
plt.tight_layout()
plt.show()


# Action Sequence Length and Command Length based on accuracy results
action_sequence_lengths = [25, 30, 35, 40, 45]
action_sequence_accuracies = [83.56, 84.51, 84.63, 83.72, 81.90]

command_lengths = [4, 6, 7, 8, 9]
command_accuracies = [82.28, 82.18, 83.30, 84.42, 83.63]

# Action Sequence Length vs. Accuracy 
plt.figure(figsize=(8, 6))
plt.bar(action_sequence_lengths, action_sequence_accuracies, width=3)
plt.title("Token-Level Accuracy by Action Sequence Length")
plt.xlabel("Ground-Truth Action Sequence Length (in words)")
plt.ylabel("Accuracy on New Commands (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Command Length vs. Accuracy 
plt.figure(figsize=(8, 6))
plt.bar(command_lengths, command_accuracies, width=0.8)
plt.title("Token-Level Accuracy by Command Length")
plt.xlabel("Command Length (in words)")
plt.ylabel("Accuracy on New Commands (%)")
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

