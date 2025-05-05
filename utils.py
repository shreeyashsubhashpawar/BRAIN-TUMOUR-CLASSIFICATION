import torch

def predict(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_gradcam(model, image_tensor, target_class, device, target_layer=None):
    model.eval()
    image_tensor = image_tensor.to(device)

    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # Use last conv layer if not specified
    if target_layer is None:
       for name, module in reversed(list(model.named_modules())):
           if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break


    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()

    class_loss = output[0, target_class]
    class_loss.backward()

    grads = gradients[0].detach()
    acts = activations[0].detach()

    weights = grads.mean(dim=[2, 3], keepdim=True)
    gradcam = torch.relu((weights * acts).sum(1, keepdim=True)).squeeze()

    gradcam = gradcam.cpu().numpy()
    gradcam = cv2.resize(gradcam, (224, 224))
    gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min())

    handle_fw.remove()
    handle_bw.remove()

    return gradcam
