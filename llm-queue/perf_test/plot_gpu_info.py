import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load the GPU information from the CSV file
df = pd.read_csv(sys.argv[1])

# Convert timestamp to datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter data based on the desired start timestamp
#start_timestamp = '06:58:13'
#df = df[df['timestamp'] >= start_timestamp]

# Convert relevant columns to float or integer
def convert_float(value):
    return float(value.split()[0])

df['memory.used'] = df['memory.used'].apply(convert_float)
df['utilization.gpu'] = df['utilization.gpu'].apply(convert_float)
df['power.draw'] = df['power.draw'].apply(convert_float)
df['temperature.gpu'] = df['temperature.gpu'].astype(int)
#print mean of utilization.gpu
print("Mean GPU Utilization: {:.2f}".format(df["utilization.gpu"].mean()))
# Set the timestamp as the index
df.set_index('timestamp', inplace=True)

# Figure 1 - Time vs Memory and Utilization
fig1, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Memory Used (MiB)', color='tab:blue')
ax1.plot(df.index, df['memory.used'], color='tab:blue', label='Memory Used')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

ax2 = ax1.twinx()
ax2.set_ylabel('GPU Utilization', color='tab:red')
ax2.plot(df.index, df['utilization.gpu'], color='tab:red', label='GPU Utilization')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig1.tight_layout()
plt.title('Memory and GPU Utilization')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
fig1.savefig(f'{sys.argv[1][:-4]}_GPU_Memory_and_Utilization.png')

# Figure 2 - Time vs Power Draw and Temperature
fig2, ax3 = plt.subplots(figsize=(10, 6))

ax3.set_xlabel('Timestamp')
ax3.set_ylabel('Power Draw', color='tab:blue')
ax3.plot(df.index, df['power.draw'], color='tab:blue', label='Power Draw')
ax3.tick_params(axis='y', labelcolor='tab:blue')
ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

ax4 = ax3.twinx()
ax4.set_ylabel('GPU Temperature', color='tab:red')
ax4.plot(df.index, df['temperature.gpu'], color='tab:red', label='GPU Temperature')
ax4.tick_params(axis='y', labelcolor='tab:red')

fig2.tight_layout()
plt.title('GPU Power Draw and Temperature')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
fig2.savefig(f'{sys.argv[1][:-4]}_GPU_Power_Draw_and_Temperature.png')
