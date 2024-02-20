# HLOP-SNN
This is the PyTorch implementation of the paper: Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks **(ICLR 2024)**. \[[openreview](https://openreview.net/forum?id=MeB86edZ1P)\] \[[arxiv](https://arxiv.org/abs/2402.11984)\]

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python`

## Training
Run as following examples:

	python spiking_train_pmnist.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -gpu-id 0
	
	# feedback alignment
	python spiking_train_pmnist.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -feedback_alignment -gpu-id 0
	
	# sign symmetric
	python spiking_train_cifar.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -sign_symmetric -gpu-id 0
	
	# baseline, i.e. vanilla sequential learning of different tasks
	python spiking_train_fivedataset.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -baseline -gpu-id 0
	
	# combination with memory replay (if combined with -baseline, corresponds to only memory replay)
	python spiking_train_fivedataset.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -replay -gpu-id 0
	
	# hlop with lateral spiking neurons
	python spiking_train_pmnist.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -hlop_spiking -hlop_spiking_scale 20. -hlop_spiking_timesteps 40 -gpu-id 0
	
	# for convolutional networks, can specify the hlop projection type for acceleration on CPU/GPU
	-hlop_proj_type weight

The default hyperparameters are the same as the paper.

## Acknowledgement

Some codes are adpated from [DSR](https://github.com/qymeng94/DSR), [OTTT](https://github.com/pkuxmq/OTTT-SNN), and [spikingjelly](https://github.com/fangwei123456/spikingjelly). Some codes for data processing are adapted from [GPM](https://github.com/sahagobinda/GPM).

## Contact

If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
