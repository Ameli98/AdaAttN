from math import inf


class fake_namespace():
    def __init__(self, content_path, style_path, result_path):
        self.content_path = content_path
        self.style_path = style_path
        self.name = "AdaAttN"
        self.gpu_ids = [0]
        self.checkpoints_dir = "./checkpoints"
        self.model = "adaattn"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 64
        self.ndf = 64
        self.netD = "basic"
        self.netG = "resnet_9blocks"
        self.n_layers_D = 3
        self.norm = "instance"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = False
        self.dataset_mode = "unaligned"
        self.direction = "AtoB"
        self.serial_batches = False
        self.num_threads = 4
        self.batch_size = 1
        self.load_size = 512
        self.crop_size = 512
        self.load_ratio = 1.0
        self.crop_ratio = 1.0
        self.max_dataset_size = inf
        self.preprocess = "resize_and_crop"
        self.no_flip = False
        self.display_winsize = 256
        self.epoch = "latest"
        self.load_iter = 0
        self.verbose = False
        self.suffix = ""
        self.results_dir = result_path
        self.phase = "test"
        self.eval = False
        self.num_test = inf
        self.image_encoder_path = "checkpoints/vgg_normalised.pth"
        self.skip_connection_3 = True
        self.shallow_layer = True
        self.isTrain = False
