import codecs
import os

train_parameters = {  
    "input_size": [3, 224, 224],  
    "class_dim": -1,  # 分類數量
    "image_count": -1,  # 訓練圖片數量 
    "label_dict": {},  
    "data_dir": "data/",  # 數據集所在位置
    "train_file_list": "train.txt",  
    "label_file": "label_list.txt",
    "save_resnet": "./best_resnet_model", 
    "save_spinalnet": "./best_spinalnet_model",  
    "continue_train": False,        # 是否接着上一次保存的參數 
    "num_epochs": 50,                #
    "train_batch_size": 256, 
    "infer_img": 'data/test/Alexandrite/alexandrite_18.jpg',
    "mean_rgb": [127.5, 127.5, 127.5],  # 三通道均值
    "image_enhance_strategy": {  # 圖像增強策略
        "need_distort": True,  # 顏色增強 
        "need_rotate": True,   # 隨機旋轉 
        "need_crop": True,      # 增加隨機裁剪
        "need_flip": True,      # 增加隨機翻轉  
        "hue_prob": 0.5,  
        "hue_delta": 18,  
        "contrast_prob": 0.5,  
        "contrast_delta": 0.5,  
        "saturation_prob": 0.5,  
        "saturation_delta": 0.5,  
        "brightness_prob": 0.5,  
        "brightness_delta": 0.125  
    },  
    "early_stop": {  
        "sample_frequency": 50,  
        "successive_limit": 3,  
        "good_acc1": 0.92  
    },  
    "rsm_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "momentum_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "sgd_strategy": {  
        "learning_rate": 0.001,  
        "lr_epochs": [20, 40, 60, 80, 100],  
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]  
    },  
    "adam_strategy": {  
        "learning_rate": 0.002  
    }  
}  

def init_train_parameters():
    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        train_parameters['class_dim'] = index
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)
