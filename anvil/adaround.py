import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class AdaRoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gamma=0.01, zeta=1.1):
        sigmoid = torch.sigmoid(alpha)
        s = (sigmoid * (zeta - gamma)) + gamma
        h_alpha = torch.clamp(s, 0, 1)
        return torch.floor(x) + h_alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def adaround_weight(weight, alpha, gamma=0.01, zeta=1.1):
    return AdaRoundFunction.apply(weight, alpha, gamma, zeta)

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

def adaround_layer(layer, inputs, num_iterations=1000, beta_range=(20, 2), reg_param=0.01, per_channel=True):
    assert isinstance(layer, nn.Conv2d), "Only Conv2d supported"

    signed = True  # Można rozpoznać np. po typie aktywacji
    qmin, qmax = (-128, 127) if signed else (0, 255)

    weight = layer.weight.detach()
    alpha = nn.Parameter(torch.zeros_like(weight))
    scale_w, zp_w = get_qparams(weight, qmin, qmax, per_channel=True, channel_axis=0)

    optimizer = torch.optim.Adam([alpha], lr=1e-2)
    best_loss = float("inf")
    best_alpha = alpha.data.clone()

    # Kwantyzacja wejścia (per-tensor)
    scale_in, zp_in = get_qparams(inputs, qmin, qmax, per_channel=False)
    inputs_q = quantize_tensor(inputs, scale_in, zp_in, qmin, qmax)

    for step in range(num_iterations):
        optimizer.zero_grad()

        # Kwantyzacja wag
        weight_q = adaround_weight(weight / scale_w + zp_w, alpha)
        weight_q = scale_w * (weight_q - zp_w)

        # Forward oryginalny vs. kwantyzowany
        out_fp = layer(inputs)
        out_q = F.conv2d(inputs_q, weight_q, bias=layer.bias, stride=layer.stride,
                         padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

        loss_data = F.mse_loss(out_q, out_fp)

        beta = beta_range[0] * (1 - step / num_iterations) + beta_range[1] * (step / num_iterations)
        h_alpha = torch.clamp(torch.sigmoid(alpha) * 1.2 - 0.1, 0, 1) #according to up and down
        reg = torch.sum(1 - torch.abs(2 * h_alpha - 1) ** beta)

        loss = loss_data + reg_param * reg
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_alpha = alpha.data.clone()

    # Finalizacja
    h_alpha = torch.clamp(torch.sigmoid(best_alpha) * 1.2 - 0.1, 0, 1)
    print(f"final h_alpha: {h_alpha}")
    final_w_q = scale_w * (torch.floor(weight / scale_w + zp_w) + h_alpha - zp_w)

    return final_w_q

# --- Wrapper ---

class AdaRoundModelWrapper:
    def __init__(self, model, sample_input):
        self.model = copy.deepcopy(model)
        self.sample_input = sample_input.to(next(self.model.parameters()).device)
        self.device = self.sample_input.device
        self.model.eval()

    def apply_adaround_to_conv_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                print(f"[AdaRound] Processing layer: {name}")

                captured_input = None

                def hook_fn(module, input, output):
                    nonlocal captured_input
                    captured_input = input[0].detach()

                hook = module.register_forward_hook(hook_fn)

                with torch.no_grad():
                    _ = self.model(self.sample_input)

                hook.remove()

                if captured_input is None:
                    raise RuntimeError(f"Nie udało się przechwycić wejścia do warstwy {name}")

                # Kwantyzacja wag
                quantized_weights = adaround_layer(module, captured_input)
                module.weight.data.copy_(quantized_weights)
                print(f"[AdaRound] -> Done.")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"[AdaRound] Quantized model saved to: {path}")
