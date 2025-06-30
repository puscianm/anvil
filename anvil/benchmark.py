# zwykły rounding
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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

def quantize_layer_weights(layer, bitwidth, per_channel):
    signed = True
    qmin, qmax = (-(2**(bitwidth - 1)), 2**(bitwidth - 1) - 1) if signed else (0, 2**bitwidth - 1)

    weight = layer.weight.data

    if isinstance(layer, nn.Conv2d):
        channel_axis = 0  # out_channels
        scale_w, zp_w = get_qparams(weight, qmin, qmax, per_channel=per_channel, channel_axis=channel_axis)
    elif isinstance(layer, nn.Linear):
        # dla Linear per_channel = False, lub można użyć axis=0 (out_features)
        scale_w, zp_w = get_qparams(weight, qmin, qmax, per_channel=per_channel)  # można też dodać parametr channel_axis=0
    else:
        raise NotImplementedError("Only Conv2d and Linear layers are supported.")

    weight_q = quantize_tensor(weight, scale_w, zp_w, qmin, qmax)
    layer.weight.data.copy_(weight_q)

# --- Wrapper ---
class StaticRoundQuantWrapper:
    def __init__(self, model, bitwidth_w, bitwidth_a):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.bitwidth_a = bitwidth_a
        self.bitwidth_w = bitwidth_w

    def apply_quant_to_layers(self, per_channel):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantize_layer_weights(module, bitwidth=self.bitwidth_a, per_channel=per_channel)

    def quantize_activations(self, sample_input):
        qmin, qmax = 0, 2**self.bitwidth_w - 1
        activation_stats = {}

        def capture_activations(name):
            def hook_fn(module, input, output):
                activation_stats[name] = output.detach()
            return hook_fn

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(capture_activations(name)))

        with torch.no_grad():
            _ = self.model(sample_input)

        for hook in hooks:
            hook.remove()

        for name, module in list(self.model.named_modules()):
            if name in activation_stats:
                activation = activation_stats[name]
                scale, zp = get_qparams(activation, qmin, qmax)

                class QuantizedReLU(nn.Module):
                    def __init__(self, scale, zp, qmin, qmax):
                        super().__init__()
                        self.scale = scale
                        self.zp = zp
                        self.qmin = qmin
                        self.qmax = qmax

                    def forward(self, x):
                        x = F.relu(x)
                        return quantize_tensor(x, self.scale, self.zp, self.qmin, self.qmax)

                parent = self.model
                modules = name.split('.')
                for m in modules[:-1]:
                    parent = getattr(parent, m)
                setattr(parent, modules[-1], QuantizedReLU(scale, zp, qmin, qmax))
    
        with torch.no_grad():
            first_layer_input = sample_input  # or capture during forward pass
            self.input_scale, self.input_zero_point = get_qparams(first_layer_input, qmin, qmax)

    def forward(self, x):
        # (NEW) Quantize input tensor before passing to the model
        if self.input_scale is not None and self.input_zero_point is not None:
            x = quantize_tensor(x, self.input_scale, self.input_zero_point, 0, 2**self.bitwidth_w - 1)
        return self.model(x)
    

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[Quant] Quantized model saved to: {path}")
