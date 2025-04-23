class Configs:
    def __init__(self):
        # Necessary parameters
        self.total_length = 22  # Total number of frames, e.g., 20
        self.input_length = 4  # Number of input frames, e.g., 10
        self.ngf = 32  # Base number of channels for the generator, e.g., 64
        self.img_height = 256  # Image height, e.g., 512
        self.img_width = 256  # Image width, e.g., 512

        # Additional parameters that may be needed
        # self.num_hidden = [64, 64, 64, 64]  # List of hidden layer channel numbers
        # self.filter_size = 5                # Size of the convolutional kernel
        # self.stride = 1                     # Stride size
        # self.layer_norm = 1                 # Whether to use layer normalization, 1 means yes
        # self.reverse_scheduled_sampling = 0 # Whether to use reverse scheduled sampling
        # self.scheduled_sampling = 1         # Whether to use scheduled sampling
        # self.sampling_stop_iter = 50000     # Number of iterations to stop scheduled sampling
        # self.sampling_start_value = 1.0     # Initial value for scheduled sampling
        # self.sampling_changing_rate = 0.00002 # Change rate for scheduled sampling

        # Input channel number for the generator
        # self.ic_feature = 8 * self.ngf * 10
        self.ic_feature = 320  # 640
        self.evo_ic = self.total_length - self.input_length
        self.gen_oc = self.total_length - self.input_length

        self.beta1 = 0.5
        self.beta2 = 0.999
        self.batch_size = 16
        self.num_epochs = 100000
        self.log_step = 100
        self.save_step = 10000
        self.sample_step = 10000
        self.model_path = "../../models"
        self.sample_path = "./samples"

        self.generation_steps: int = 6  # 6 in paper

        self.discriminator_type = "conv3d"
        # self.discriminator_type = 'dgmr'
        if self.discriminator_type == "conv3d":
            self.pool_reg_weight = 10.0
            self.g_lr = 0.0002
            self.d_lr = 0.0002
        elif self.discriminator_type == "dgmr":
            self.g_lr = 0.0002
            self.d_lr = 0.00005
            self.precip_weight_cap = 64.0
        else:
            raise ValueError(f"Unknown discriminator type: {self.discriminator_type}")
