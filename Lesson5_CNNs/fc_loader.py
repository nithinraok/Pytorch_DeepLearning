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

def train(Model,trainloader,testloader,criterion,optimizer,epochs):
    test_losses,train_losses=[],[]
    for e in range(epochs):
        running_loss=0;

        Model.train();
        for images,labels in trainloader:
            images_t = images.view(images.shape[0],-1);
            optimizer.zero_grad();

            logits=Model.forward(images_t);
            loss_t=criterion(logits,labels);
            loss_t.backward();
            optimizer.step();

            running_loss+=loss_t;

        else:
            test_loss,accuracy=validation(Model,testloader,criterion);

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy))

        test_losses.append(test_loss/len(testloader))
        train_losses.append(running_loss/len(trainloader))

    return train_losses,test_losses,accuracy

def validation(Model,testloader,criterion):
    test_loss=0;
    accuracy=0;

    Model.eval();
    images_num=0;
    with torch.no_grad():
        for images,labels in testloader:
            images_num+=images.shape[0];
            images_t=images.view(images.shape[0],-1);
            logits=Model.forward(images_t);
            loss_t=criterion(logits,labels)
            test_loss+=loss_t;

            _,pred_labels=torch.topk(logits,1,dim=1)
            equality=(labels==pred_labels.view(*labels.shape))

            accuracy += torch.sum(equality)

    overall_acc=accuracy.float()/images_num;
    return test_loss,overall_acc

def load_model(filepath):
    checkpoint=torch.load(filepath)

    Model = Network(checkpoint['input_size'],
                    checkpoint['output_size'],
                    checkpoint['hidden_layers'],
                    dropout_p=0.2)
    Model.load_state_dict(checkpoint['state_dict'])

    return Model
