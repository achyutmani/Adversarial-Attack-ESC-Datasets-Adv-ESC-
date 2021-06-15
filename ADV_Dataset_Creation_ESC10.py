# Acoustic Event Detection Using Knowledge Distillation from Attention-Based Subband Specilist Deep Model 
import torch.optim as optim # Import optim 
import torchvision # Import torchvision 
import h5py #import h5py
from torch.utils.data import dataset # import dataset
from ESC10Customdataset_Train import LAEData_Train # Call Customdataloader to read Training Data
from ESC10Customdataset_Test import LAEData_Test # Call Customdataloader to read Test Data
from torch.utils.data import DataLoader # Import Dataloader 
import torchvision.transforms as transforms # Import Transform 
import pandas as pd # Import Pnadas 
import torch # Import Torch 
import torch.nn as nn # Import NN module from Torch 
from torchvision.transforms import transforms# Import transform module from torchvision 
from torch.utils.data import DataLoader # Import dataloader from torch 
from torch.optim import Adam # import optimizer module from torch 
from torch.autograd import Variable # Import autograd from torch 
import numpy as np # Import numpy module 
import torchvision.datasets as datasets #Import dataset from torch 
from Attention import PAM_Module # import position attention module 
#from Attention import CAM_Module # import channel attention module
#from Attention import SA_Module # Import Self attention module
from torch import optim, cuda # import optimizer  and CUDA
import random # import random 
import torch.nn.functional as F # Import nn.functional 
import time # import time 
import sys # Import System 
import os # Import OS
from pytorchtools import EarlyStopping
from torchvision import models
import warnings
from sklearn.metrics import confusion_matrix
SEED = 1234 # Initialize seed 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda') # Define device type 
num_classes=10 # Define Number of classes 
in_channel=1   # Define Number of Input Channels 
learning_rate=2e-4 # Define Learning rate 
batch_size=64 # Define Batch Size 
EPOCHS =1000   # Define maximum Number of Epochs
FC_Size=512
SFC_Size=512
Temp=3
alpha=0.7
N_models=6
warnings.filterwarnings("ignore")
train_transformations = transforms.Compose([ # Training Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
test_transformations = transforms.Compose([ # Test Transform 
    #transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])])
train_dataset=LAEData_Train(transform=train_transformations) # Create tensor of training data 
Test_Dataset=LAEData_Test(transform=test_transformations)# Create tensor of test dataset 
train_size = int(0.7 * len(train_dataset)) # Compute size of training data using (70% As Training and 30% As Validation)
valid_size = len(train_dataset) - train_size # Compute size of validation data using (70% As Training and 30% As Validation)
Train_Dataset,Valid_Dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size]) # Training and Validation Data After (70%-30%)Data Split 
#train_set,test_set=torch.utils.data.random_split(dataset,[6000,2639])
#Labels=pd.read_csv("Devlopment.csv")
train_loader=DataLoader(dataset=Train_Dataset,batch_size=batch_size,shuffle=True) # Create Training Dataloader 
valid_loader=DataLoader(dataset=Valid_Dataset,batch_size=batch_size,shuffle=False)# Create Validation Dataloader 
test_loader=DataLoader(dataset=Test_Dataset,batch_size=batch_size,shuffle=False) # Create Test Dataloader 
class Teacher(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
    def __init__(self):
        super(Teacher, self).__init__()
        Pre_Trained_Layers=nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-2])
        #Pre_Trained_Layers = list(models.resnet34(pretrained=True).children())[:-4]
        #Pre_Trained_Layers = models.resnet34(pretrained=True) # Initialize model layers and weights
        #print(Pre_Trained_Layers)
        self.features=Pre_Trained_Layers
        self.PAM=PAM_Module(512)
        #self.CAM=CAM_Module(512)
        self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #self.features.Flat=nn.Flatten()
        self.fc=nn.Linear(512,num_classes)  # Set output layer as an one output

    def forward(self,image):
        x = self.features(image)
        #x1=self.PAM(x)
        #x2=self.CAM(x1)
        #x3=x1+x2
        x4=self.avgpool(x)
        #x4=x3.view(x3.shape[0],-1)
        x4=x4.view(x4.size(0),-1)
        #x4=torch.flatten(x4)
        #print(x4.shape)
        #x4=torch.unsqueeze(x4,-1)
        #print(x4.shape)
        x5=self.fc(x4)
        return x5
Teacher_Model=Teacher()
#print(Teacher_Model)
Teacher_Model=Teacher_Model.to(device)
Teacher_optimizer = optim.Adam(Teacher_Model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss() # Define Loss Function
import torchattacks
#attack =torchattacks.DeepFool(Teacher_Model,2)
#attack =torchattacks.DeepFool(Model1,5)
#attack =torchattacks.DeepFool(Model1,10)
attack=torchattacks.FGSM(Teacher_Model,eps=32/255)
#attack=torchattacks.FGSM(Model1,eps=0.5)
#attack=torchattacks.FGSM(Model1,eps=1)
#attack=torchattacks.BIM(Teacher_Model, 32/255,1/255,0)
#attack=torchattacks.BIM(Model1, 1/255,16/255,0)
#attack=torchattacks.BIM(Model1, 1/255,32/255,0)
#attack= torchattacks.PGD(Teacher_Model,16/255,1/255)# In case if accuracy is coming same try to change the secind parameter 1/255 to other 8/255, 16/255,32/255
#attack= torchattacks.PGD(Model1,16/255,1/255)
#attack= torchattacks.PGD(Model1,32/255,1/255)
#attack= torchattacks.PGD(Teacher_Model, 1/255, 1/255, 40,True)# Here you can change secnd and third parameters for more details see github repo of torch attacks
#attack= torchattacks.PGD(Model1, 1/255, 16/255, 40,True)
#attack= torchattacks.PGD(Model1, 1/255, 32/255, 40,True)
#attack =torchattacks.DeepFool(resnet_model,3)
#attack=torchattacks.FGSM(resnet_model,eps=0.3)
# attack=torchattacks.FGSM(model, eps=0.007)
#attack=torchattacks.BIM(resnet_model, 1/255,16/255,0)
# attack = torchattacks.StepLL(model, eps=4/255, alpha=1/255, steps=0)
#attack= torchattacks.RFGSM(model, eps=16/255, alpha=32/255)
#attack= torchattacks.CW(resnet_model, targeted=False, c=1, kappa=0, steps=1000, lr=0.01)
#attack= torchattacks.PGD(model, eps=16/255, alpha=32/255)
#attack= torchattacks.PGD(resnet_model, 1/255, 1/255, 40,False)
# attack = torchattacks.DeepFool(model, steps=10)
#attack= torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
#attack= torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=7)

def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    #T=3
    #LL=nn.KLDivLoss()((F.log_softmax(fx/T,dim=1)),(F.softmax(fx/T,dim=1)))
    #print(fx.shape)
    return acc
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
def calculate_accuracy(fx, y): # caluate accuracy 
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = correct.float()/preds.shape[0]
    return acc
def ADV_Dataset(Data,key):
    N_Sigs=len(key)
    key=np.array(key)
    #Data=Data.cpu()
    with h5py.File('ADV_ESC10_PGD_S3_Test.hdf5', 'w') as f:
        for i in range(0,len(Data)):
            SG_Data=Data[i]
            SG_Data=torch.squeeze(SG_Data)
            SG_Data=SG_Data.cpu()
            SG_Data=SG_Data.numpy()
            print(i)
            #print(SG_Data.shape)
            f.create_dataset(key[i], data=SG_Data)
        #print(SG_Data)

def evaluate(model,device,iterator, criterion): # Evaluate Validation accuracy 
    #print("Validation Starts")
    epoch_loss = 0
    epoch_acc = 0
    count=0
    model.eval() # call model object for evaluation 
    #with torch.no_grad(): # Without computation of gredient 
    for (x,y,key) in iterator:
        x=x.float()
        x=x.to(device) # Transfer data to device 
        y=y.to(device) # Transfer label  to device 
        count=count+1
        adv_images=attack(x,y)
        ADV_Dataset(adv_images,key)
        adv_images=adv_images.to(device)
        Predicted_Label = model(adv_images) # Predict claa label 
        loss = criterion(Predicted_Label, y) # Compute Loss 
        acc = calculate_accuracy(Predicted_Label, y) # compute Accuracy 
        #print("Validation Iteration Number=",count)
        epoch_loss += loss.item() # Compute Sum of  Loss 
        epoch_acc += acc.item() # Compute  Sum of Accuracy 
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator) 

SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join("/media/mani/Ph.D./PHD_Journals/AK/Adversarial Attacks Defense Using GAN/Adversarial Attack Datasets", 'ESC10_CNN.pt') # Define Path to save the model 
Teacher_Model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss, test_acc = evaluate(Teacher_Model, device, test_loader, criterion)
