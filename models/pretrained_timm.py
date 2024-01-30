import torch as t
import torch.nn as nn


class PretrainedTimmNet(nn.Module):
    
    def __init__(
        self,
        inmodel,
        numberOfLinearLayers,
        dropOutLin,
        intermediateLinearLayerShape,
        linearActivationFunc,
        sigmoidOn,
        yShape,
    ) -> None:
        super().__init__()

        self.modelName = "pretrained_timm_net"
        self.numberOfLinearLayers = numberOfLinearLayers
        self.dropOutLin = dropOutLin
        self.intermediateLinearLayerShape = intermediateLinearLayerShape
        self.linearActivationFunc = linearActivationFunc
        self.sigmoidOn = sigmoidOn
        if 'test_input_size' in inmodel.pretrained_cfg.keys():
            with t.no_grad():
                a = t.rand(inmodel.pretrained_cfg['test_input_size']).unsqueeze(0)
                outshape = inmodel(a).shape
            self.linInShape = outshape[1]
        else:
            self.linInShape = inmodel.norm.normalized_shape[0]

        if len(yShape)>1:
            self.outShape = yShape[1]
        else:
            self.outShape = yShape[0]

        self.vit_main = inmodel    

        linearLayers = []
        layerInShape = self.linInShape
        for linLayer in range(1,numberOfLinearLayers):
            linearLayers.append(nn.Dropout(dropOutLin))
            linearLayers.append(nn.Linear(layerInShape,intermediateLinearLayerShape))
            layerInShape = intermediateLinearLayerShape
            if linLayer < numberOfLinearLayers:
                if linearActivationFunc != "Linear":
                    linearLayers.append(getattr(nn,linearActivationFunc)())
        linearLayers.append(nn.Dropout(dropOutLin))
        linearLayers.append(nn.Linear(layerInShape,self.outShape))

        if sigmoidOn:
            linearLayers.append(nn.Sigmoid())

        self.linModel = nn.Sequential(*linearLayers)
    
    def forward(self,x):
        #mostly from https://github.com/rwightman/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/vision_transformer.py
        x = self.vit_main(x)

        linOut = self.linModel(x)
        return linOut
    
    def remove_final_linear(self):
        self.linModel = nn.Identity()