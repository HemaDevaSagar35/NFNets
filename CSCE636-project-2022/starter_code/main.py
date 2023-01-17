### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs
from ImageUtils import visualize, parse_record


def configure():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str, default='train', help="hypertune, train, test or predict")
	parser.add_argument("--data_dir", type=str, help="path to the data")
	parser.add_argument("--save_dir", type=str, default = '../', help="path to save the results")

	return parser.parse_args()

def hyper_tune(model_configs, data_dir, save_dir):
	x_train, y_train, x_test, y_test = load_data(data_dir)
	x_train_new, y_train_new, x_valid, y_valid = train_valid_split(x_train, y_train)
	params = {}
	params['lr'] = [0.001, 0.01, 0.1]
	params['batch_size'] = [32, 64, 128, 256]
    # we are only hyper tuning on batch size and learning rate
	epochs = 40
	save_interval = 10
	output_file = os.path.join(save_dir, 'output_log.txt')
	os.makedirs(save_dir, exist_ok=True)
	print("###### started training in the hyper space #####")
	for lr in params['lr']:
		for batch_size in params['batch_size']:
			train_configs = {
				'batch_size':batch_size,
				'learning_rate':lr,
				'save_interval':save_interval,
				'max_epoch':epochs}
			model = MyModel(model_configs)
			val_results = model.train(x_train_new, y_train_new, train_configs, x_valid, y_valid)
			with open(output_file, "a") as writer:
				writer.write("Validation accuracies for lr {}, batch size {} at variou epoch is {}\n".format(
					lr, batch_size, val_results
					)
					)

if __name__ == '__main__':
	configs = configure()

	print(model_configs)
	if configs.mode == 'hypertune':
		hyper_tune(model_configs, configs.data_dir, configs.save_dir)

	else:
		model = MyModel(model_configs)
		if configs.mode == 'train':
			#print(training_configs)

			output_file = os.path.join(configs.save_dir, 'final_train_log.txt')
			x_train, y_train, x_test, y_test = load_data(configs.data_dir)
			x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

			results = model.train(x_train, y_train, training_configs, x_valid, y_valid)
			test_acc = model.evaluate(x_test, y_test)

			with open(output_file, "a") as writer:
				writer.write("final log accuracies for lr {}, batch size {} at variou epoch is {}\n".format(
					training_configs['learning_rate'], training_configs['batch_size'], results
					)
					)
				writer.write("Test Accuracy at the end is {}".format(test_acc))

			
		elif configs.mode == 'test':
			# Testing on public testing dataset
			_, _, x_test, y_test = load_data(configs.data_dir)
			# if you only want to run test model with the best saved model, you have to load the model first.
			final_model = os.path.join(model_configs['save_dir'], 'model-200.ckpt')
			model.load(final_model)
			print(model.evaluate(x_test, y_test)) # putting print to print it on console.

		elif configs.mode == 'predict':
			# Loading private testing dataset
			x_test = load_testing_images(configs.data_dir)
			# visualizing the first testing image to check your image shape
			visualize(x_test[0], 'test.png')
			# Predicting and storing results on private testing dataset
			# Again here we have to load the checkpoint first
			
			final_model = os.path.join(model_configs['save_dir'], 'model-200.ckpt')
			model.load(final_model)

			### My laptop is having issues with prediction probabilities and running into OOM (out of memory)error
			### So I am break the input into batches and remerging the prediction
			batch_size = 64
			num_batches = x_test.shape[0] // batch_size
			rem = 1 if x_test.shape[0] % batch_size > 0 else 0
			predictions = []
			for i in range(num_batches+rem):
				start_batch = i*batch_size 
				end_batch = (i+1)*batch_size
				
				x_batch = x_test[start_batch:end_batch]
				x_batch = np.array(list(map(lambda x_i : parse_record(x_i, False), x_batch)))
				
				preds = model.predict_prob(x_batch)
				predictions.append(preds.detach().cpu().numpy())
				
			predictions = np.concatenate(predictions, axis=0)

			np.save(os.path.join(configs.save_dir, 'predictions.npy'), predictions)
		

### END CODE HERE

