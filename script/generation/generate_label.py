import sys
import os
import h5py
import numpy as np
import json
import joblib

from models.normalizations import instance_local_stand
from sklearn.preprocessing import MinMaxScaler

from labeling import *


def generate_label(dataset_name, device, fullframe, predictor, byte, normalization, save_folder_data, save_folder_norm, split):
    in_file  = h5py.File(dataset_name, "r")

    learn_traces = np.array(in_file['LEARN'][device][:, fullframe])
    attack_traces = np.array(in_file['ATTACK'][device][:, fullframe])

    learn_raw_plaintexts = in_file['LEARN']['data']['plaintext']
    learn_raw_keys = in_file['LEARN']['data']['key']
    learn_raw_masks = in_file['LEARN']['data']['masks']

    attack_raw_plaintexts = in_file['ATTACK']['data']['plaintext']
    attack_raw_keys = in_file['ATTACK']['data']['key']
    attack_raw_masks = in_file['ATTACK']['data']['masks']

    learn_data = in_file['LEARN']['data']
    attack_data = in_file['ATTACK']['data']

    learn_labels = np.array(labels[predictor](learn_raw_plaintexts, learn_raw_keys, learn_raw_masks, byte))
    attack_labels = np.array(labels[predictor](attack_raw_plaintexts, attack_raw_keys, attack_raw_masks, byte))

    if normalization == "minmax":
        scaler_learn = MinMaxScaler(feature_range=(-1, 1))
        learn_traces = scaler_learn.fit_transform(learn_traces)

        scaler_attack = MinMaxScaler(feature_range=(-1, 1))
        attack_traces = scaler_attack.fit_transform(attack_traces)

        # save scaler
        joblib.dump(scaler_learn, save_folder_norm+'scaler_learn.pkl')
        joblib.dump(scaler_attack, save_folder_norm+'scaler_attack.pkl')
        
        print('Scalers saved into %s' % save_folder_norm)
    
    elif normalization == "instance_local_stand":
        learn_traces = instance_local_stand(learn_traces)
        attack_traces = instance_local_stand(attack_traces)
    elif normalization == "max":
        learn_traces = learn_traces/np.amax(learn_traces) #normalization
        attack_traces = attack_traces/np.amax(attack_traces) #normalization
    else:
        raise NotImpelementedError

    if device == "08BNANOPW":
        print('Treat Nanopw')
        print(attack_traces.shape)
        attack_traces = attack_traces[0:25000]
        print(attack_traces.shape)

    # Device - Size - Prediction - Byte
    np.save(save_folder_data+"X_profiling.npy", learn_traces[0:int(split*learn_traces.shape[0])])
    np.save(save_folder_data+"Y_profiling.npy", learn_labels[0:int(split*learn_labels.shape[0])])
    np.save(save_folder_data+"Metadata_profiling.npy", learn_data[0:int(split*learn_data.shape[0])])
    np.save(save_folder_data+"X_attack.npy", attack_traces)
    np.save(save_folder_data+"Y_attack.npy", attack_labels)
    np.save(save_folder_data+"Metadata_attack.npy", attack_data)
    in_file.close()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('> python generate_label_stm32.py [CONFIG]')
        sys.exit(1)

    config_json = sys.argv[1]
    with open(config_json) as json_file:
        json_data = json.load(json_file)

        data = json_data["dataset"]
        root = data["root"]
        dataset_name = data["name"]
        size_frame = data["size_frame"]
        frame = data["frame"]
        predictor = data["predictor"]
        byte = data["byte"]
        root_folder = data["root_folder_data"]
        normalization = data["normalization"]
        split = data["split"]

        if len(frame) != len(split):
            print("[Error] The size of split is not the same as the number of frame!")
            sys.exit(1)


        for device, rate in zip(frame, split):
            frame_device = frame[device]
            lt = []
            frame_size = 0
            
            for f in frame_device:
                f0 = f[0]
                f1 = f[1]

                r0 = list(range(f0, f1, 1))
                lt = lt + r0

            frame_size = len(lt)

            if frame_size == size_frame:

                save_folder_data = root_folder + "data/"+device+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
                if not os.path.exists(save_folder_data):
                    os.makedirs(save_folder_data)
                    print(save_folder_data+" was created! ")

                save_folder_norm = root_folder + "norm/"+device+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
                if not os.path.exists(save_folder_norm):
                    os.makedirs(save_folder_norm)
                    print(save_folder_norm+" was created! ")

                generate_label(root+dataset_name, device, lt, predictor, byte, normalization, save_folder_data, save_folder_norm, rate)
            else:
                print(frame_size, size_frame)
                print('Error: the size of the frame is not the same as defined!')