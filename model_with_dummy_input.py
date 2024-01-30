from models.pretrained_timm import PretrainedTimmNet
import timm
import torch as t


cfg = dict(
    x_shape_enc = (3,224,224), # (seqlen,c,h,w)
    num_targets = 1,
    use_first_embedding = True,
    loop_over_timesteps = True,
    num_lin_layers = 1,
    intermediate_linear_layer_shape = 512,
    dropout_lin_layer = 0.1,
    add_sigmoid = False,
    linear_activation_func_vit = "SiLU",
    vit_pretrained_name= "deit_tiny_patch16_224",
)

model = timm.create_model(
    cfg["vit_pretrained_name"],
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)

model = PretrainedTimmNet(
    model,
    yShape=(cfg["num_targets"],),
    numberOfLinearLayers=cfg['num_lin_layers'],
    dropOutLin=cfg["dropout_lin_layer"],
    intermediateLinearLayerShape=cfg["intermediate_linear_layer_shape"],
    sigmoidOn = cfg["add_sigmoid"],
    linearActivationFunc=cfg["linear_activation_func_vit"],
)


x_in = t.rand(1,3,224,224)

with t.inference_mode():
    out = model(x_in)

print(f"""Irradiance estimation would be
       {out.numpy()}""")