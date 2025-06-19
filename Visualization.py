import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from FullFwdPropagation import full_forward_propagation
from GetAccuracyValue import get_accuracy_value
from itertools import islice
from Test import validation_fidelity_r2

def visualize_NN_results(X_train, Y_train, params_values, nn, min_data, max_data, outputIndexEntry, loss_history, X_valid, Y_valid):

    m = Y_train.shape[0]
    dataset = np.arange(0, m, 1)

    network_layers = dict(islice(nn.items(), 1, None))
    Y_hat, _ = full_forward_propagation(X_train, params_values, network_layers)
    fidelity_score = get_accuracy_value(Y_hat, Y_train, min_data, max_data, outputIndexEntry, threshold=3)
    
    Y_hat_valid, _ = full_forward_propagation(X_valid, params_values, network_layers)
    total_fidelity = validation_fidelity_r2(Y_valid, Y_hat_valid)

    # Scrive i valori di Y_hat e Y_train su un file di testo TrainHatData.txt
    with open('TrainHatData.txt', 'w') as f:
        for i in range(Y_hat.shape[0]):  # Itera sulle righe
            # Alterna le colonne: Y_hat[:,0], Y_train[:,0], Y_hat[:,1], Y_train[:,1], ...
            row = []
            for j in range(Y_hat.shape[1]):  # Itera sulle colonne
                row.append(Y_hat[i, j])  # Aggiungi Y_hat corrente
                row.append(Y_train[i, j])  # Aggiungi Y_train corrente
            f.write("\t\t".join(map(str, row)) + "\n")  # Scrive la riga alternata

    # Estrae 4000 righe randomiche dal file di testo TrainHatData.txt e le scrive su un 
    # altro file di testo TrainHatData_reduced.txt mantenendo invariata la prima riga di intestazione
    with open('TrainHatData.txt', 'r') as f:
        lines = f.readlines()  
        header = lines[0]     
        data_lines = lines[1:] 
    np.random.shuffle(data_lines)
    selected_lines = data_lines[:4000]
    selected_lines.insert(0, header)
    with open('TrainHatData_reduced.txt', 'w') as f:
        f.writelines(selected_lines)

    # Calcola i valori di left_value e right_value per le bande di errore del 5%
    # right_value_low = np.zeros(len(outputIndexEntry))
    # left_value_low = np.zeros(len(outputIndexEntry))
    # right_value_hig = np.zeros(len(outputIndexEntry))
    # left_value_hig = np.zeros(len(outputIndexEntry))
    # i = 0
    # for j in outputIndexEntry:
    #     right_value_low[i] = (0.99*max_data[j]-min_data[j]) / (max_data[j]-min_data[j])
    #     left_value_low[i]  = (0.99*min_data[j]-min_data[j]) / (max_data[j]-min_data[j])
    #     right_value_hig[i] = (1.01*max_data[j]-min_data[j]) / (max_data[j]-min_data[j])
    #     left_value_hig[i]  = (1.01*min_data[j]-min_data[j]) / (max_data[j]-min_data[j])
    #     i += 1
    # combined = [left_value_hig,
    #             right_value_hig,
    #             left_value_low,
    #             right_value_low]
    # # Scrivi combined su file di testo
    # with open('combined.txt', 'w') as f:
    #     for i in range(len(combined[0])):
    #         row = []
    #         for j in range(len(combined)):
    #             row.append(combined[j][i])
    #         f.write("\t\t".join(map(str, row)) + "\n")


    # # Create a figure
    fig = plt.figure(figsize=(10, 10))
    # # Calculate the number of rows for the subplots
    # n = nn["OutputLayer"]["output_dim"]
    # rows = np.ceil(n / 2).astype(int)

    # # Loop over the range and create subplots
    # for index in range(n):
    #     ax = fig.add_subplot(rows, 2, index+1)
        
    #     # Intervallo esterno: rosso con trasparenza
    #     ax.fill_between(dataset, -15, -10, color='red', alpha=0.4)
    #     ax.fill_between(dataset, 10, 15, color='red', alpha=0.4)
        
    #     # Intervallo centrale: blu con trasparenza
    #     ax.fill_between(dataset, -10, -3, color='blue', alpha=0.4)
    #     ax.fill_between(dataset, 3, 10, color='blue', alpha=0.4)
        
    #     # Intervallo interno: verde con trasparenza
    #     ax.fill_between(dataset, -3, 3, color='green', alpha=0.4)
        
    #     ax.scatter(dataset, diff_percent[:, index], label="Error %", s=2, edgecolors='black', linewidths=0.5)
    #     ax.legend()
        
    #     # Set the title for each subplot
    #     ax.set_title(f'Plot {index+1}')

    # # If the number of subplots is odd, remove the last (empty) subplot
    # if n % 2 != 0:
    #     fig.delaxes(fig.axes[-1])

    # # Adjust the layout and save the figure
    # plt.tight_layout()
    # plt.savefig('all_plots.png')


    # for index in range(n):
    #     ax = fig.add_subplot(rows, 2, index + 1)
        
    #     ax.scatter(Y_train[:, index], Y_hat[:, index], label='Predicted', s=1)
    #     ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='red')
    #     ax.set_xlim([0, 1])
    #     ax.set_ylim([0, 1])
    #     ax.legend()
        
    #     # Set the title for each subplot
    #     ax.set_title(f'Plot {index + 1}')

    # # If the number of subplots is odd, remove the last (empty) subplot
    # if n % 2 != 0:
    #     fig.delaxes(fig.axes[-1])

    # # Adjust the layout and save the figure
    # plt.tight_layout()
    # plt.savefig('fig2.png')

    fig, ax = plt.subplots()
    index = 1
    # ax.minorticks_on()
    # ax.grid(which='both')
    # Intervallo esterno: rosso con trasparenza
    # ax.fill_between(dataset, -15, -10, color='red', alpha=0.4)
    # ax.fill_between(dataset, 10, 15, color='red', alpha=0.4)
    # # Intervallo centrale: blu con trasparenza
    # ax.fill_between(dataset, -10, -3, color='blue', alpha=0.4)
    # ax.fill_between(dataset, 3, 10, color='blue', alpha=0.4)
    # # Intervallo interno: verde con trasparenza
    # ax.fill_between(dataset, -3, 3, color='green', alpha=0.4)
    # ax.scatter(dataset, diff_percent[:, 1-1], label="Predicted", s=2, edgecolors='black', linewidths=0.5)
    # ax.legend()
    # ax.set_xlim([0, m])
    # ax.set_ylim([-15, 15])
    # ax.set_title('Error % CFL')
    # ax.set_xlabel('Dataset')
    # ax.set_ylabel('Error %')
    # plt.savefig('fig1.png')

    # fig, ax = plt.subplots()
    # index = 1
    # ax.minorticks_on()
    # ax.grid(which='both')
    # ax.scatter(Y_train[:, index], Y_hat[:, index], label='Predicted', s=1)
    # ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='red')
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    # ax.set_xlabel('Normalised Real Values')
    # ax.set_ylabel('Normalised Predicted Values')
    # ax.set_title('Predicted vs Real Values of Normalised CFL')
    # ax.legend()
    # plt.savefig('fig2.png')

    # indices = np.argsort(Y_train[:, 5])
    # Y_valid_sorted = Y_train[indices]
    # Y_hat_sorted = Y_hat[indices]
    
    # err_per_variable    = np.mean(diff_percent, axis=0)
    # mean_errors         = ', '.join(f"{err:.2f}%" for err in err_per_variable)
    # sigma_per_variable  = np.sqrt(np.var(diff_percent, axis=0).astype(float))
    # sigma_errors        = ', '.join(f"{var:.2f}" for var in sigma_per_variable)

    # Plotting the loss function
    # fig, ax = plt.subplots()
    # ax.plot(loss_history)
    # ax.set_yscale('log')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Loss')
    # ax.set_title('Loss function')
    # ax.grid(which='both')
    # plt.savefig('fig3.png')

    # fig, ax = plt.subplots()
    # ax.scatter(dataset,Y_hat_sorted[:, 5], label='Predicted', s=1)
    # ax.plot(Y_valid_sorted[:, 5], label='Real', color='red')
    # ax.legend()
    # plt.show()
   
    return