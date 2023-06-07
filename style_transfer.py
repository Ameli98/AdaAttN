from data import create_dataset
from models import create_model
from config import fake_namespace
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm


def main(content_path, style_path, result_path):
    opt = fake_namespace(content_path, style_path, result_path)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.serial_batches = True
    # no flip; comment this line if results on flipped images are needed.
    opt.no_flip = True
    # no visdom display; the test code saves the results to a HTML file.
    opt.display_id = -1
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    # create a model given opt.model and other options
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)

    result_dir = Path(result_path)
    if opt.eval:
        model.eval()
    for data in tqdm(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img = visuals["cs"]
        img_path = model.get_image_paths()[0]     # get image paths
        save_image(img, result_dir / img_path)

    # Pack the image folder
    for img_path in sorted(result_dir.iterdir()):

        style_folder_name, filename = str(img_path).rsplit("_", 1)
        style_folder = Path(style_folder_name)
        if not style_folder.exists():
            style_folder.mkdir()
        img_path.rename(style_folder / filename)


if __name__ == "__main__":
    main("temp/Yes", "style", "./results")
