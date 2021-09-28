Parameters for experimentation
--------------------------------
depth_search = [12, 14, 16, 18, 20]
width_search = [4, 6, 8, 10, 12]
coarse_classes range(19)
crop_size

Stage 1 Zoo:
Transform data
Set fine classes to coarse classes
predicting range(0 - 19) labels, with 20 neurons
Export

Stage 2 Zoo: 
Fine classes remain unchanged
output 100 neurons, 100 labels
Split the dataset by the coarse labels (20 subsets)
training towards fine labels

vegetable (superclass) <--
- apples
- lettuce
- tomato
- carrot
- beans

----

Yanbo Notes

- L73 in cifar100 use torch.save to export dataset as tensors before the dataloader is init, do not save loader
- TODO! Write down a list of all experiment permutations for a tracking todo