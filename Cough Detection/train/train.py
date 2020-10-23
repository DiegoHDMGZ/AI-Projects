from fastai.vision import *
from pathlib import Path
from fastai.callbacks import *

class Agent:
    def setPathSave(self, path):
        self.pathSave = path
    
    def readData(self, path):
        self.data = (ImageList.from_folder(path ) 
                    .split_by_folder(train='Training', valid='Test')  
                    .label_from_folder()            
                    .transform(self.tfms , size=256)    
                    .databunch(bs = 32)) 
                     
    def showData(self):
        self.data.show_batch(3, figsize=(6,6), hide_axis=False)
    
    def createNeuralNetwork(self):
        self.net = cnn_learner(self.data, models.resnet34, metrics=[error_rate , accuracy])
    
    def showLearningRateGraphic(self, minLearningRate, maxLearningRate):
        self.net.lr_find(minLearningRate ,  maxLearningRate)
        self.net.recorder.plot()
    
    def train(self, epochs, minLearningRate, maxLearningRate):
        self.net.fit_one_cycle(epochs, max_lr=slice(minLearningRate, maxLearningRate), 
                               callbacks=[SaveModelCallback(self.net, every='improvement', monitor = 'accuracy', name= self.pathSave/'model')])
    
    def showConfusionMatrix(self):
        interp = ClassificationInterpretation.from_learner(self.net)
        interp.plot_confusion_matrix()
    
    def showMostConfused(self):
        interp = ClassificationInterpretation.from_learner(self.net)
        interp.plot_top_losses(9, figsize=(15,11))
     
