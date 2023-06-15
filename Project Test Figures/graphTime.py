import pandas as pd
import matplotlib.pyplot as plt

# Read the first file
file1 = pd.read_csv('Bachelor/time_taking.csv')
file1 = file1[file1.notnull().sum(axis=1) == 3]  # Filter rows with three values

# Read the second file
file2 = pd.read_csv('P6-Project/Algorithm/time_taking.csv')
file2 = file2[file2.notnull().sum(axis=1) == 3]  # Filter rows with three values

plt.plot(range(len(file1)), file1.iloc[:, 1], label='5 Features')
plt.plot(range(len(file2)), file2.iloc[:, 1], label='7 Features')
plt.xlabel('Frame', fontsize=20)
plt.ylabel('Execution time', fontsize=20)

# Add dashed lines and labels
plt.axhline(y=0.1, color='red', linestyle='dashed', label='10 FPS')
plt.axhline(y=0.5, color='blue', linestyle='dashed', label='2 FPS')

plt.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()