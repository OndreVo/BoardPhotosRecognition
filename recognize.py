from fastai.vision import *
from fastai.metrics import error_rate
from os import listdir, rename, path
from os.path import isfile, join
from multiprocessing import freeze_support
import ctypes
import ntpath

ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

bs = 16 #batch size - bigger is better but runs out of memory
relearn = False # change to True when you want to relearn model - otherwise it will use stored model
learn_data_folder = 'path to a folder with learning data'
folder_to_process = 'path to a folder with new photos'
export_file = 'model.pkl'


def get_boardrcg_model():   
    model = None 
    if relearn or not path.exists(join(learn_data_folder, export_file)):        
        print("training model ...")
        data = ImageDataBunch.from_folder(learn_data_folder, ds_tfms=get_transforms(), size=224, bs=bs)
        # using convolutional neural network and Transfer Learning (use pretrained model resnet34 as base)
        model = cnn_learner(data, models.resnet34, metrics=error_rate)
        model.fit_one_cycle(4)
        model.export(file=export_file)
    else:
        print("loading model ...")
        model = load_learner(learn_data_folder, file=export_file)
    return model


def move_bycat(filepath, model):
    ## recognize cat
    img = open_image(filepath)
    cat,_,_ = model.predict(img)
    ## move file
    print(cat, end=' ')
    filename = ntpath.basename(filepath)
    rename(filepath, join(folder_to_process, join(str(cat), filename)))
    pass


def process_folder(model):
    ## walk images in the folder    
    print("processing images ...")
    files_list = [join(folder_to_process,f) for f in listdir(folder_to_process) if '.jpg' in f]
    i = 0
    l = len(files_list)
    for filepath in files_list:
        i += 1
        print('\r',i,'/',l,ntpath.basename(filepath),'... ', end=' ')
        move_bycat(filepath, model)
    
    

if __name__ == '__main__':
    freeze_support()
    model = get_boardrcg_model()
    process_folder(model)

