import torch
import torch.nn as nn
import torchvision.models as models

class FineTuneVGG16(nn.Module):
    def __init__(self, model_path, num_classes):
        super(FineTuneVGG16, self).__init__()
        # # load VGG model 
        original_model = models.vgg16()
        # print(model)
        original_model.load_state_dict(torch.load(model_path))

        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

        # print(original_model)

        # since we are using the pre-trained model as a feature extractor we set
        # its parameters to non-trainable (do not calculate gradient to update weights)
        for param in self.features.parameters():
            param.requires_grad = False
                
        # print(model)
        last_item_index = len(original_model.classifier) - 1
        old_fc = original_model.classifier.__getitem__(last_item_index)
        new_fc = nn.Linear(in_features=old_fc.in_features, out_features= num_classes, bias=True)
        self.classifier.__setitem__(last_item_index , new_fc)
        
        # only update last layer
        for i in range(last_item_index):
            for param in self.classifier[i].parameters():
                param.requires_grad = False
        
    def forward(self, t):
        input = t
        t = self.features(input)
        t = self.avgpool(t)
        t = torch.tensor(t)
        t = t.view(t.size(dim=0), -1)
        print(t.shape)
        out = self.classifier(t)
        return out

        
