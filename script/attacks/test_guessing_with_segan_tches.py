import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from plot_guessing_entropy import check_file_exists, load_sca_model
from plot_guessing_entropy import full_ranks_perso


# Check a saved model against one of the ASCAD databases Attack traces
def check(list_model_file, X_attack, Y_attack, Metadata_attack, num_traces, predictor, use_segan=False, filename_gan=None):

    if len(X_attack.shape) == 2:
        nb_chans = 1
    else:
        nb_chans = X_attack.shape[2]

    print(nb_chans)

    if use_segan:
        z_input_shape = (2, 256)
        z = np.random.normal(0, 1, size=(X_attack.shape[0], ) + z_input_shape)

        # load the GAN model
        generator = load_model(filename_gan) # gaussian

        X_attack = generator.predict([X_attack, z])


    result = []
    for i in range(0, len(list_model_file)):
        model_file = list_model_file[i]
        check_file_exists(model_file)

        # Load model
        model = load_sca_model(model_file)

        list_mean = []

        for j in range(0, X_attack.shape[0], num_traces):
            print(j, '->', j+num_traces)
            X_attack_selected = X_attack[j:j+num_traces, :]
            Metadata_attack_selected = Metadata_attack[j:j+num_traces]

            ranks = full_ranks_perso(model, X_attack_selected, Metadata_attack_selected, 0, num_traces, 1, predictor)

            # We plot the results
            x = [ranks[i][0] for i in range(0, ranks.shape[0])]
            y = [ranks[i][1] for i in range(0, ranks.shape[0])]
            list_mean.append(y)
        
        y = np.array(list_mean).mean(axis=0)
        result.append(y)
    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('> python src/attack2/test_guessing_wtih_segan_tches.py [CONFIG]')
        sys.exit(1)

    config_json = sys.argv[1]
    with open(config_json) as json_file:
        json_data = json.load(json_file)
        data_dataset = json_data['dataset']
        data_attack = json_data['attack']

    dataset_name = data_dataset["name"]
    size_frame = data_dataset["size_frame"]
    predictor = data_dataset["predictor"]
    byte = data_dataset["byte"]
    root_folder_data = data_dataset["root_folder_data"]
    normalization = data_dataset["normalization"]

    channel_target = data_attack['target']
    #zoom = data_attack['zoom']
    #graph_name = data_attack['graph_name']
    folder_numeric_graph = data_attack['folder_numeric_graph']
    folder_png_graph = data_attack['folder_png_graph']
    
    if not os.path.exists(folder_numeric_graph):
        os.makedirs(folder_numeric_graph)
        print(folder_numeric_graph+" was created! ")

    if not os.path.exists(folder_png_graph):
        os.makedirs(folder_png_graph)
        print(folder_png_graph+" was created! ")

    metric_attack = data_attack['metric_attack']

    if dataset_name.startswith("traces_O0_3"):
        num_traces = 200
    elif dataset_name.startswith("trace_ASCAD-protected"):
        num_traces = 10000
    elif dataset_name.startswith("trace_BD"):
        num_traces = 1000
    else:
        raise NotImplementedError


    list_channel = []
    nb_channels = len(channel_target.split('*'))
    for channels in channel_target.split('*'):
        load_data = root_folder_data + "data/"+channels+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
        
        X1 = np.load(load_data+"X_attack.npy")
        Y_attack = np.load(load_data+"Y_attack.npy")
        Metadata_attack = np.load(load_data+"Metadata_attack.npy")
        list_channel.append(X1)


    X_attack = np.dstack(list_channel)
    print(X_attack.shape)

    #fig1 = plt.figure()
    fig, axs = plt.subplots(2)
    fig.suptitle('Target '+channel_target)
    plt.xlabel('number of traces', fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('mean rank', fontsize=14)
    plt.grid(True)
    #plt.legend(prop={"size": 12})

    # for the basique models
    for method in json_data['training'].keys():
        print(method)
        data_training = json_data['training'][method]

        if (method == "multichannel") or (method == "multichannel2") or (method == "transfer"):
            data = data_training["data"]
            root_folder_models = data_training["root_folder_models"]
            nb_kfold = data_training["kfold"]
            ntp = data_training["ntp"]

            for train_on in data:

                subfolder = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + train_on
                #subfolder = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_1_" + train_on
                #trained_models_folder = root_folder_models + subfolder + "/"
                trained_models_folder = root_folder_models + subfolder + "/models/"+metric_attack+"/"

                models = []

                list_all_files1 = os.listdir(trained_models_folder)
                list_all_files1.sort()

                list_model_only1 = []
                print(list_all_files1[-1])
                list_model_only1.append(list_all_files1[-1])
   
                #for files in list_all_files1:
                 #   if files.endswith("_best-loss.hdf5"):
                  #      list_model_only1.append(files)

                for list_model in list_model_only1:
                    models.append(trained_models_folder+list_model)

                values = check(models, X_attack, Y_attack, Metadata_attack, num_traces, predictor)
                print(values)
                print(len(values))

                if train_on == channel_target:
                    color='grey'
                    style='solid'
                elif ("+" in train_on):
                    color='blue'
                    style='solid'
                elif train_on.count("*") == 1:
                    color='green'
                    style='solid'
                elif train_on.count("*") == 2:
                    color='purple'
                    style='solid'
                else:
                    color='red'
                    style='solid'

                for y in values:
                    #if zoom:
                    axs[0].plot(y,color=color, label=train_on, linestyle=style)
                    axs[1].plot(y[0:30],color=color, label=train_on, linestyle=style)
                    
                    axs[0].legend()
                    axs[1].legend()

                    #else:
                    np.save(folder_numeric_graph+train_on+'.npy', y)
                    



                if ("gan" in data_attack.keys()) and ("generative" in json_data['training'].keys()): # if gan is available
                    temp_gan = channel_target+"+"+train_on

                    gan_use = data_attack["gan"]["gan_use"]
                    gan_selected = data_attack["gan"]["gan_selected"]

                    if temp_gan in gan_use: # if the current channel was treat by gan
                        if temp_gan in json_data['training']["generative"]["data"]: # if the training exist
                            print(temp_gan)
                            print(models)

                            root_folder_gan = json_data['training']["generative"]['root_folder_models']
                            nb_kfold_gan = json_data['training']["generative"]['kfold']

                            subfolder_gan = str(nb_kfold_gan) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + "1" + "_" + temp_gan
                            trained_gan_folder = root_folder_gan + subfolder_gan + "/"

                            values = check(models, X_attack, Y_attack, Metadata_attack, num_traces, predictor, use_segan=True, filename_gan=trained_gan_folder+gan_selected)
                            print(values)

                            #plt.title('Target '+channel_target)
                            #plt.xlabel('number of traces', fontsize=14)
                            #plt.yticks(fontsize=14)
                            #plt.ylabel('mean rank', fontsize=14)
                            #plt.grid(True)

                            if temp_gan == channel_target:
                                color='grey'
                                style='solid'
                            elif "+" in temp_gan:
                                color='blue'
                                style='solid'
                            else:
                                color='red'
                                style='solid'

                            for y in values:
                                #if zoom:
                                axs[0].plot(y,color=color, label=temp_gan, linestyle=style)
                                axs[1].plot(y[0:10],color=color, label=temp_gan, linestyle=style)
                                #else:
                                np.save(folder_numeric_graph+temp_gan+'.npy', y)
                                #plt.legend(prop={"size": 12})

    #if zoom:
    #fig1.savefig(folder_png_graph+"graph_"+metric_attack+".png")
    #fig2.savefig(folder_png_graph+"graph_"+metric_attack+"_zoom.png")
    #else:
    plt.savefig(folder_png_graph+"graph_"+metric_attack+".png")
    plt.clf()