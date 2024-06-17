import paddle.fluid as fluid
from ResNet import ResNet
import numpy as np
import paddle
import reader
import os
import logging
from config import train_parameters, init_train_parameters
import time

def init_log_config():  

    global logger  
    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)  
    log_path = os.path.join(os.getcwd(), 'logs')  
    if not os.path.exists(log_path):  
        os.makedirs(log_path)  
    log_name = os.path.join(log_path, 'train.log')  
    sh = logging.StreamHandler()  
    fh = logging.FileHandler(log_name, mode='w')  
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")  
    fh.setFormatter(formatter)  
    sh.setFormatter(formatter)  
    logger.addHandler(sh)  
    logger.addHandler(fh)  

def optimizer_momentum_setting(parameter_list):  

    learning_strategy = train_parameters['momentum_strategy']  
    batch_size = train_parameters["train_batch_size"]  
    iters = train_parameters["image_count"] // batch_size  
    lr = learning_strategy['learning_rate']  
  
    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]  
    values = [i * lr for i in learning_strategy["lr_decay"]]  
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)  
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, parameter_list=parameter_list)  
    return optimizer  
  
  
def optimizer_rms_setting(parameter_list):  

    batch_size = train_parameters["train_batch_size"]  
    iters = train_parameters["image_count"] // batch_size  
    learning_strategy = train_parameters['rsm_strategy']  
    lr = learning_strategy['learning_rate']  
  
    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]  
    values = [i * lr for i in learning_strategy["lr_decay"]]  
  
    optimizer = fluid.optimizer.RMSProp(  
        learning_rate=fluid.layers.piecewise_decay(boundaries, values), parameter_list=parameter_list)  
  
    return optimizer  
  
  
def optimizer_sgd_setting(parameter_list):  

    learning_strategy = train_parameters['sgd_strategy']  
    batch_size = train_parameters["train_batch_size"]  
    iters = train_parameters["image_count"] // batch_size  
    lr = learning_strategy['learning_rate']  
  
    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]  
    values = [i * lr for i in learning_strategy["lr_decay"]]  
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)  
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate, parameter_list=parameter_list)  
    return optimizer  
  
  
def optimizer_adam_setting(parameter_list):  

    learning_strategy = train_parameters['adam_strategy']  
    learning_rate = learning_strategy['learning_rate']  
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate, parameter_list=parameter_list)  
    return optimizer  

def eval_net(reader, model):
    acc_set = []
    
    for batch_id, data in enumerate(reader()):
        dy_x_data = np.array([x[0] for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).astype('int')
        y_data = y_data[:, np.newaxis]
        img = fluid.dygraph.to_variable(dy_x_data)
        label = fluid.dygraph.to_variable(y_data)
        label.stop_gradient = True
        prediction, acc = model(img, label)
        
        acc_set.append(float(acc.numpy()))

        # get test acc and loss
    acc_val_mean = np.array(acc_set).mean()

    return acc_val_mean  
    
def train():
    
    with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
        all_train_rewards=[]
        all_eval_rewards=[]
        epoch_num = train_parameters["num_epochs"]
        net = ResNet("resnet", class_dim = train_parameters['class_dim'])
        optimizer = optimizer_rms_setting(net.parameters())
        file_list = os.path.join(train_parameters['data_dir'], "train.txt")
        train_reader = paddle.batch(reader.custom_image_reader(file_list, train_parameters['data_dir'], 'train'),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=True)
        test_reader = paddle.batch(reader.custom_image_reader(file_list, train_parameters['data_dir'], 'val'),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=True)
        if train_parameters["continue_train"]:
            model, _ = fluid.dygraph.load_dygraph(train_parameters["save_resnet"])
            net.load_dict(model)
            
        best_acc = 0
        for epoch_num in range(epoch_num):
            
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([x[1] for x in data]).astype('int')
                y_data = y_data[:, np.newaxis]
                
                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True
                t1 = time.time()
                out, acc = net(img, label)
                t2 =time.time()
                forward_time = t2 - t1
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)
                # dy_out = avg_loss.numpy()
                t3 = time.time()
                avg_loss.backward()
                t4 = time.time()
                backward_time = t4 - t3
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                # print(forward_time, backward_time)
                
                dy_param_value = {}
                for param in net.parameters():
                    dy_param_value[param.name] = param.numpy

                if batch_id % 40 == 0:
                   logger.info("Loss at epoch {} step {}: {}, acc: {}".format(epoch_num, batch_id, avg_loss.numpy(), acc.numpy()))
                   all_train_rewards.append(acc.numpy()[0])     
                   net.eval()
                   eval_acc = eval_net(test_reader, net)
                   all_eval_rewards.append(eval_acc)
                   net.train()
                   if  eval_acc > best_acc:
                        fluid.dygraph.save_dygraph(net.state_dict(), train_parameters["save_resnet"])
                        best_acc = eval_acc
                        logger.info("model saved at epoch {}, best accuracy is {}".format(epoch_num, best_acc))
        logger.info("Final loss: {}".format(avg_loss.numpy()))
        np.savez('result/resnet_result.npz', all_train_rewards=all_train_rewards, all_eval_rewards=all_eval_rewards)


if __name__ == "__main__":
    init_log_config()
    init_train_parameters()
    train()