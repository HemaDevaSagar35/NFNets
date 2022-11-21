### YOUR CODE HERE
# import tensorflow as tf
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')
torch.manual_seed(3)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3)

class MyModel(nn.Module):

    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.network = MyNetwork(
            self.config['block_size'],
            self.config['classes'],
            self.config['first_num_filters']
        ).to(DEVICE)
       
        #self.lr = self.config.lr
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), 
                                         lr=0.01, 
                                         momentum=0.9,
                                         weight_decay = self.config['weight_decay'])

    # def model_setup(self):
    #     pass
    # I definitely changd the argument catalog here, for ease of mu implementation
    def train(self, x_train, y_train, train_config, x_valid=None, y_valid=None):

        for g in self.optimizer.param_groups:
            g['lr'] = train_config['learning_rate']
        print('learning rate is {}'.format(g['lr']))
        num_samples = x_train.shape[0]
        num_batches = num_samples // train_config['batch_size']
        rem = 1 if num_samples % train_config['batch_size'] > 0 else 0
        print('### Training... ###')
        val_accuracies = {}
        for epoch in range(1, train_config['max_epoch']+1):
            self.network.train()
            start_time = time.time()
            # Shuffle
            #np.random.seed(3)
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            running_loss = 0
            if epoch % 50 == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = g['lr']/10
            
            for i in range(num_batches + rem):
               
                start_batch = i*train_config['batch_size']
                end_batch = (i+1)*train_config['batch_size']
                
                x_batch = curr_x_train[start_batch:end_batch]
                y_batch = curr_y_train[start_batch:end_batch]
                
                x_batch = np.array(list(map(lambda x : parse_record(x, True), x_batch)))
                
                x_batch = torch.tensor(x_batch).float().to(DEVICE)
                y_batch = torch.tensor(y_batch).long().to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.network(x_batch)
                
                loss = self.loss(outputs, y_batch)
                running_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                #print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss))
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            # if x_valid is not None:
            #     val_error = evaluate(self, x_valid, y_valid)
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, running_loss/num_batches, duration))
            # if x_valid is not None:
            #     print("Val error is {}".format(val_error))

            if epoch % train_config['save_interval'] == 0:
                self.save(epoch)
                val_accuracies[epoch] = [running_loss/num_batches]

                if x_valid is not None:
                    val_acc = self.evaluate(x_valid, y_valid)
                    val_accuracies[epoch].append(val_acc)
                    print("validation accuracy after epoch {} is {}".format(epoch, val_acc))
                    
        return val_accuracies

    def evaluate(self, x, y):
        self.network.eval()
        print('### Evaluate on evaluation set ###')
        accuracy_dict = {}

        batch_size = 64
        num_batches = x.shape[0] // batch_size
        rem = 1 if x.shape[0] % batch_size > 0 else 0
        preds = []

        for i in tqdm(range(num_batches+rem)):
            #print(end_batch)
            start_batch = i*batch_size 
            end_batch = (i+1)*batch_size
            
            x_batch = x[start_batch:end_batch]
            y_batch = y[start_batch:end_batch]

            x_batch = np.array(list(map(lambda x_i : parse_record(x_i, False), x_batch)))
            prediction = self.predict_prob(x_batch)
            #x_batch = torch.tensor(x_batch).float().cuda()
            #prediction = self.network(x_batch)
            #prediction = torch.argmax(prediction, dim=1)
            preds.extend(prediction)
        
        y = torch.tensor(y)
        preds = torch.tensor(preds)
        acc = torch.sum(preds==y)/y.shape[0]
        print('Test accuracy: {:.4f}'.format( round(acc.item(),4)))
        
        return round(acc.item(),4)


    def predict_prob(self, x):
        ### right this assumes there is cuda present in the platform be sure to mention about
        x = torch.tensor(x).float().to(DEVICE)
        prediction = self.network(x)
        prediction = torch.argmax(prediction, dim=1)
        return prediction
    
    def save(self, epoch):
        checkpoint_path = os.path.join(self.config['save_dir'], 'model-%d.ckpt'%(epoch))
        os.makedirs(self.config['save_dir'], exist_ok=True)
        torch.save(self.network.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location="cpu")
        self.network.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))


### END CODE HERE