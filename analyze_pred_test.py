import matplotlib.pyplot as plt
import pandas as pd
import sys
def plot_pred_test(input_csv, output_dir):

    #read  csv
    df_power = pd.read_csv(input_csv)
    #plot y_test vs y_pred
    fig, ax = plt.subplots()
    #calculate mean squared error
    
    df_power['error'] = (df_power['y_pred']-df_power['y_test'])

    
    ax.scatter(df_power['y_test'], (df_power['error']),)

    #output rows with df_power['y_pred']-df_power['y_test'] < 0
    under_pred = df_power[df_power['error'] >0]
    under_pred = under_pred.sort_values(by='error', ascending=False)

    #
    #sort df_power by df_power['y_pred']-df_power['y_test']
    df_power = df_power.sort_values(by='y_pred', ascending=False)
    #print top 10 rows of df_power
    print(under_pred.head(20))
    ax.set_xlabel("y_test")
    ax.set_ylabel("y_pred-y_test")
    ax.set_title("y_test vs y_pred")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/y_test_vs_y_pred.png")
    
input_csv = sys.argv[1]
import os
#get input dir from input_csv
input_csv = str(os.path.abspath(input_csv))
output_dir = "/".join(input_csv.split("/")[:-1])
plot_pred_test(input_csv=input_csv, output_dir=output_dir)
    