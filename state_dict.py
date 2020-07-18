import torch
import models
import torch.nn as nn
model = models.WRN(depth=32, width=10, num_classes=10).cuda()
model = models.convert_splitbn_model(model).cuda()
#model = nn.DataParallel(model).cuda()
ckpt_path = "./results/train/spbn_best.pth"
model_data = torch.load(ckpt_path)
model.load_state_dict(model_data['model'])
#for name, param in model.named_parameters():
    #print(name, param.requires_grad)

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        print(m)
        m.reset_parameters()
        m.eval()
def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()
        for name, param in  module.named_parameters():
            print(name, param.requires_grad)

def set_bn_to_eval(m):
    classname = m.__class__.__name
    if classname.find('BatchNorm') != -1:
        m.eval()

def freeze_bn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            print( 'freeze: ' + name, module)
            for names, modules in module.named_children():
                modules.train()
                print( 'unfreeze: ' + names)
        else:
            freeze_bn(module)

freeze_bn(model)
#if you want to set_bn_to_eval of some subnet or some basenetwork then just use.
#model.apply(set_bn_to_eval)

#model.apply(set_bn_eval)


#for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #if param_tensor=="module.group0.block0.bn0.running_mean":
    #    print(model.state_dict()[param_tensor].mean())
    #if 'aux' in param_tensor:
        #rint(param_tensor)
        #print("{:.4f}".format(float(model.state_dict()[param_tensor].mean())))
