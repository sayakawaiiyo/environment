"""
Prediction file
"""

import re
import os
import time
import torch
import argparse
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from RGBBranch import RGBBranch
from SemBranch import SemBranch
from SASceneNet import SASceneNet
from Libs.Datasets.SUN397Dataset import SUN397Dataset
from Libs.Utils import utils
from Libs.Utils.torchsummary import torchsummary
import numpy as np
import yaml


#parser = argparse.ArgumentParser(description='Semantic-Aware Scene Recognition Evaluation')
#parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')

def find_string_by_number(filename, search_number):
    try:
        with open(filename, 'r',encoding='utf-8') as file:
            for line in file:
                parts = line.split()
                if len(parts) == 2:
                    string, number = parts
                    number = int(number)
                    if number == search_number:
                        return string
            return None  # Number not found in the file
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None

def evaluationDataLoader(dataloader, model, set):

    Predictions = np.zeros(len(dataloader))

    # Extract batch size
    batch_size = CONFIG['VALIDATION']['BATCH_SIZE']['TEST']

    with torch.no_grad():
        for i, (mini_batch) in enumerate(dataloader):
            if USE_CUDA:
                RGB_image = mini_batch['Image'].cuda()
                semantic_mask = mini_batch['Semantic'].cuda()
                semantic_scores = mini_batch['Semantic Scores'].cuda()

            if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                # Fuse batch size and ncrops to set the input for the network
                bs, ncrops, c_img, h, w = RGB_image.size()
                RGB_image = RGB_image.view(-1, c_img, h, w)

                bs, ncrops, c_sem, h, w = semantic_mask.size()
                semantic_mask = semantic_mask.view(-1, c_sem, h, w)

                bs, ncrops, c_sem, h, w = semantic_scores.size()
                semantic_scores = semantic_scores.view(-1, c_sem, h, w)

            # Create tensor of probabilities from semantic_mask
            semanticTensor = utils.make_one_hot(semantic_mask, semantic_scores, C=CONFIG['DATASET']['N_CLASSES_SEM'])

            # Model Forward
            outputSceneLabel, feature_conv, outputSceneLabelRGB, outputSceneLabelSEM = model(RGB_image, semanticTensor)

            if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                # Average results over the 10 crops
                outputSceneLabel = outputSceneLabel.view(bs, ncrops, -1).mean(1)


            if batch_size is 1:
                if set is 'Validation' and CONFIG['VALIDATION']['TEN_CROPS']:
                    feature_conv = torch.unsqueeze(feature_conv[4, :, :, :], 0)
                    RGB_image = torch.unsqueeze(RGB_image[4, :, :, :], 0)

                # Obtain 10 most scored predicted scene index
                Ten_Predictions = utils.obtainPredictedClasses(outputSceneLabel)

                # Save predicted label and ground-truth label
                Predictions[i] = Ten_Predictions[0]

        print("")
        #print(np.transpose(Predictions))
        return np.transpose(Predictions)


global USE_CUDA, classes, CONFIG

CONFIG = yaml.safe_load(open('Config/config_SUN397.yaml', 'r'))
USE_CUDA = torch.cuda.is_available()

def environment_predict(jpg_path):
    print('-' * 65)
    print("Prediction starting ...")
    print('-' * 65)

    # Instantiate network
    model = SASceneNet(arch=CONFIG['MODEL']['ARCH'], scene_classes=CONFIG['DATASET']['N_CLASSES_SCENE'], semantic_classes=CONFIG['DATASET']['N_CLASSES_SEM'])

    # Load the trained model
    # index1 = model_path.find("Data/")
    #
    # if index1 != -1:
    #     # 提取 "Data/" 之后的部分
    #     model_path_out = model_path[index1:]
    #completePath = './' + model_path_out
    completePath = CONFIG['MODEL']['PATH'] + CONFIG['MODEL']['NAME'] + '.pth.tar'
    #print(completePath)
    if os.path.isfile(completePath):
        checkpoint = torch.load(completePath)
        model.load_state_dict(checkpoint['state_dict'])

    if USE_CUDA:
        model.cuda()
    cudnn.benchmark = USE_CUDA
    model.eval()

    valdir = os.path.join(CONFIG['DATASET']['ROOT'], CONFIG['DATASET']['NAME'])

    index2 = jpg_path.find("val/")

    if index2 != -1:
        # 提取 "val/" 之后的部分
        jpg_path_out = jpg_path[index2:]
    val_dataset = SUN397Dataset(valdir,jpg_path_out, tencrops=CONFIG['VALIDATION']['TEN_CROPS'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['VALIDATION']['BATCH_SIZE']['TEST'],
                                                 shuffle=False, num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], pin_memory=True)

    # Check if OUTPUT_DIR exists and if not create it
    if not os.path.exists(CONFIG['EXP']['OUTPUT_DIR']):
        os.makedirs(CONFIG['EXP']['OUTPUT_DIR'])

    # Evaluate model on validation set
    val_prdeictions = evaluationDataLoader(val_loader, model, set='Validation')

    # Input string
    input_string = str(val_prdeictions)

    # Remove brackets, split the string by whitespace, and convert to floats
    number = float(input_string.replace('[', '').replace(']', ''))

    # Define the filename
    scene_names_cn = 'Data/Datasets/SUN397/scene_names_cn.txt'
    result = find_string_by_number(scene_names_cn, number)

    print('图片'+ jpg_path + '的标签为'+ result)

    return result


#environment_predict("val/park/sun_aapqlhldddojcstw.jpg")










