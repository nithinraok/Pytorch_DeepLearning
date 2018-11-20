import torch
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):

    def __init__(self,n_input,n_output,hidden_layers,dropout_p=0.3):
        super().__init__()

        layer_sizes = zip(hidden_layers[:-1],hidden_layers[1:])

        self.hidden_layers = nn.ModuleList([nn.Linear(n_input,hidden_layers[0])])

        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1],n_output)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self,x):
        for each in self.hidden_layers:
            x=each(x);
            x=F.relu(x);
            x=self.dropout(x);

        x=self.output(x);
        x=F.log_softmax(x,dim=1);

        return x;

def train(Model,trainloader,testloader,criterion,optimizer,epochs=5):

    train_losses,test_losses=[],[]
    for e in range(epochs):
        running_loss=0;
        Model.train()
        for images,labels in iter(trainloader):
            t_images=images.view(images.shape[0],-1)
            logits=Model.forward(t_images)
            optimizer.zero_grad()
            loss = criterion(logits,labels)
            loss.backward();
            optimizer.step();

            running_loss+=loss.item();

        else:

            test_losses,accuracy= validation(Model,testloader,criterion)
            Model.train(); # This ensures to add back dropout layer for train

            print("Epoch: {}/{}".format(e,epochs),"Train Loss: {:.3f} ".format(running_loss/len(trainloader)),
                  "Test_Loss: {:.3f}".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

def validation(Model,testloader,criterion):
    test_loss=0;
    accuracy=0;

    with torch.no_grad():
        Model.eval(); # This ensures to remove dropout layer for test
        for images,labels in iter(testloader):

            t_images=images.view(images.shape[0],-1)
            output=Model(t_images)
            pred=torch.argmax(output,dim=1)
            accuracy+=torch.mean((pred==labels).type(torch.FloatTensor))

            loss = criterion(output,labels);
            test_loss+=loss.item();

    test_losses.append(test_loss/len(testloader))
    train_losses.append(running_loss/len(trainloader))

    return test_loss,accuracy
