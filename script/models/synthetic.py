import json
import sys
import h5py
import numpy as np
from tensorflow.keras.models import load_model


def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return


def generate_fake(model_filename, gan_model, load_data, channels, save_fake):
	# Load attack data
	X_attack = np.load(load_data+"X_attack.npy")
	Y_attack = np.load(load_data+"Y_attack.npy")
	Metadata_attack = np.load(load_data+"Metadata_attack.npy")


	g_model = load_model(model_filename)

	if gan_model == "segan":
		z_input_shape = (2,256)
		z = np.random.normal(0, 1, size=(X_attack.shape[0], ) + z_input_shape)
		fake_attack_dataset = g_model.predict([X_attack, z])
	elif gan_model == "pix2pix":
		fake_attack_dataset = g_model.predict([X_attack])
	else:
		raise NotImpelementedError

	fake_attack_dataset = fake_attack_dataset.reshape((fake_attack_dataset.shape[0], fake_attack_dataset.shape[1], 1))

	file = h5py.File(save_fake, "w")
	file.create_dataset('FAKE/data', data=Metadata_attack)
	file.create_dataset('FAKE/traces_'+channels, data=fake_attack_dataset)
	file.create_dataset('FAKE/labels', data=Y_attack)
	file.close()



if __name__ == "__main__":

	if len(sys.argv) < 2:
		print('> python src/models/main.py [CONFIG]')
		sys.exit(1)

	config_json = sys.argv[1]
	with open(config_json) as json_file:
		json_data = json.load(json_file)
		data_dataset = json_data['dataset']

	dataset_name = data_dataset["name"]
	size_frame = data_dataset["size_frame"]
	predictor = data_dataset["predictor"]
	byte = data_dataset["byte"]
	root_folder_data = data_dataset["root_folder_data"]
	normalization = data_dataset["normalization"]

	if "training" in json_data.keys():
		if "generative" in json_data['training'].keys():
			data_training = json_data['training']["generative"]
			#method = data_training["method"]

			data = data_training["data"]
			root_folder_model = data_training["root_folder_models"]
			nb_kfold = data_training["kfold"]
			ntp = data_training["ntp"]
			gpu = data_training["gpu"]
			arch = data_training["arch"]
			gan_model = data_training["gan_model"]
			gan_selected = data_training["gan_selected"]

			for d in data:
				data_noisy, data_clean = d.split("+")				
				load_data = root_folder_data + "data/"+data_noisy+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"

				subfolder_gan = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + "1" + "_" + d
				trained_gan_folder = root_folder_model + subfolder_gan + "/"
				save_fake = "./fake_"+d+".h5"
				print(">> Saved: ", save_fake)
				generate_fake(trained_gan_folder+gan_selected, gan_model, load_data, data_noisy, save_fake)
		else:
			print("[Error] The config file need to have a 'generative' dict into 'training'!")
			sys.exit(1)
	else:
		print("[Error] The config file need to have a 'training' dict!")
		sys.exit(1)
