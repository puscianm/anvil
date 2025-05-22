import torch
import copy
import torch.nn as nn

def get_qparams(tensor, qmin, qmax, per_channel=False, channel_axis=0):
    if per_channel:
        dims = list(range(tensor.ndim))
        dims.remove(channel_axis)
        min_vals = tensor.amin(dim=dims, keepdim=True)
        max_vals = tensor.amax(dim=dims, keepdim=True)
    else:
        min_vals = tensor.min()
        max_vals = tensor.max()
    scale = (max_vals - min_vals) / float(qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = torch.round(qmin - min_vals / scale)
    return scale, zero_point

def quantize_tensor(tensor, scale, zero_point, qmin, qmax):
    q = torch.round(tensor / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return scale * (q - zero_point)

def quantize_layer_weights(layer, per_channel=True):
    assert isinstance(layer, nn.Conv2d), "Only Conv2d supported"
    signed = True
    qmin, qmax = (-128, 127) if signed else (0, 255)

    weight = layer.weight.data
    scale_w, zp_w = get_qparams(weight, qmin, qmax, per_channel=per_channel, channel_axis=0)
    weight_q = quantize_tensor(weight, scale_w, zp_w, qmin, qmax)

    layer.weight.data.copy_(weight_q)
    print("[Quant] Weights quantized.")

# --- Wrapper ---
class StaticRoundQuantWrapper:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
        self.model.eval()

    def apply_quant_to_conv_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"[Quant] Processing layer: {name}")
                quantize_layer_weights(module)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[Quant] Quantized model saved to: {path}")
