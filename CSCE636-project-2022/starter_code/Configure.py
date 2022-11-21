# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"classes" : 10,
	"block_size" : 5,
	"first_num_filters" : 16,
	"weight_decay" : 2e-4

	# ...
}

training_configs = {
	"learning_rate": 0.01,
	"batch_size": 64,
	"max_epoch":200,
	"save_interval" : 10
}


### END CODE HERE