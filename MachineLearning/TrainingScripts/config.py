trainroot=r'train_gen'
testroot=r'test_gen'
output_dir = 'output/'

gpu_id = 0
workers = 64
start_epoch = 0
epochs = 100

train_batch_size = 8
back_step = 10

lr = 1e-3
end_lr = 1e-7
lr_decay = 0.1
lr_decay_step = 50
display_interval = 100
use_compile = True
restart_training = True
checkpoint = ''

# random seed
seed = 2

# Mixed Precision Training
use_amp = True  # Sử dụng Automatic Mixed Precision

# Gradient Clipping
grad_clip = 1.0  # Max gradient norm (0 để tắt)

# Gradient Accumulation
accumulation_steps = 4

# Evaluation
eval_interval = 5

# Display settings  
display_interval = 10  # Log mỗi 10 batches