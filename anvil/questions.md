# Activation function error 

We don't account for error introduced by activation function mentioned in equality (25) in up and down

# Calculation of the loss function
In our code now we have
```
loss_data = F.mse_loss(out_q, out_fp)
```

This is theoretically not frobenius norm as we devide by n but the outcome should be the same.