<h1 align="center"><b>SAM Optimizer</b></h1>
<h3 align="center"><b>Sharpness-Aware Minimization for Efficiently Improving Generalization</b></h1>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 
 
--------------

<br>

Unofficial implementation of [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412).


## Usage

```python
from sam import SAM
...

model = YourModel()
optimizer = SAM(model.parameters())
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
