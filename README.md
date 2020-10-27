<h1 align="center"><b>SAM Optimizer</b></h1>
<h3 align="center"><b>Sharpness-Aware Minimization for Efficiently Improving Generalization</b></h1>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 
 
--------------

<br>

SAM simultaneously minimizes loss value and loss sharpness. In particular, it seeks parameters that lie in neighborhoods having uniformly low loss. SAM improves model generalization and yields novel SoTA performance for several datasets. Additionally, it provides robustness to label noise on par with that provided by SoTA procedures that specifically target learning with noisy labels.

This is an unofficial repository for [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412). Implementation-wise, SAM class is a light wrapper that computes the regularized "sharpness-aware" gradient that is used by the underlying optimizer (such as SGD with momentum). This repository also includes a simple WRN for Cifar10; as a proof-of-concept, it beats the performance of SGD with momentum on this dataset.


## Usage

```python
from sam import SAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
...

for input, output in data:

  # first forward-backward pass
  loss = loss_function(output, model(input))  # use this loss for any training statistics
  loss.backward()
  optimizer.first_step(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward()
  optimizer.second_step(zero_grad=True)
...
```

## Documentation

#### `SAM.__init__`

| **Argument**    | **Description** |
| :-------------- | :-------------- |
| `params` (iterable) | iterable of parameters to optimize or dicts defining parameter groups |
| `base_optimizer` (torch.optim.Optimizer) | underlying optimizer that does the "sharpness-aware" update |
| `rho` (float, optional)           | size of the neighborhood for computing the max loss *(default: 0.05)* |
| `**kwargs` | keyword arguments passed to the `__init__` method of `base_optimizer` |

<br>

#### `SAM.first_step`

Performs the first optimization step that finds the weights with the highest loss in the local `rho`-neighborhood.

<br>

#### `SAM.second_step`

Performs the second optimization step that updates the original weights with the gradient from the (locally) highest point in the loss landscape.