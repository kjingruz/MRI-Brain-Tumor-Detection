from IPython.display import clear_output
import warnings
import random
import cv2
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from matplotlib import pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances

warnings.filterwarnings("ignore")

!pip install 'git+https://github.com/facebookresearch/detectron2.git'
clear_output()




class MyTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, COCOEvaluatorHook(
            self.cfg.DATASETS.TEST[0],
            self.cfg,
            True,
            self.model,
            self.optimizer,
            self.scheduler
        ))
        return hooks

class COCOEvaluatorHook(HookBase):
    def __init__(self, dataset_name, cfg, is_final, model, optimizer, scheduler):
        self._dataset_name = dataset_name
        self._cfg = cfg
        self._is_final = is_final
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._best_metric = -1

    def after_step(self):
        self.trainer.storage.put_scalar('learning_rate', self._optimizer.param_groups[0]["lr"], smoothing_hint=False)
        if self.trainer.iter % self._cfg.TEST.EVAL_PERIOD == 0 or self._is_final:
            self.trainer.model.eval()
            results = inference_on_dataset(self._model, build_detection_test_loader(self._cfg, self._dataset_name), COCOEvaluator(self._dataset_name, self._cfg, True))
            self.trainer.model.train()
            if results['bbox']['AP'] > self._best_metric:
                self._best_metric = results['bbox']['AP']
                # Here you can implement saving the best model
            else:
                # Stop training if metric hasn't improved
                raise EarlyStoppingException()

class EarlyStoppingException(Exception):
    pass


register_coco_instances("my_dataset_train", {}, "./train/_annotations.coco.json", "./train")
register_coco_instances("my_dataset_val", {}, "./valid/_annotations.coco.json", "./valid")


# Set up the logger
setup_logger()

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.DEVICE = "cpu"

# Set up data augmentation
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.9, 0.9]
cfg.INPUT.RANDOM_FLIP = "horizontal"

# Set up learning rate scheduler
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.WARMUP_METHOD = "linear"
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
cfg.SOLVER.GAMMA = 0.05

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Evaluation with COCO Evaluator
evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
inference_on_dataset(trainer.model, val_loader, evaluator)


def get_class_name_from_filename(filename):
    if 'Tr-gl' in filename:
        return 'Glioma'
    elif 'Tr-me' in filename:
        return 'Meningioma'
    elif 'Tr-pi' in filename:
        return 'Pituitary'
    elif 'Tr-no' in filename:
        return 'No Tumor'
    else:
        return 'Unknown'

# Inference with the trained model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

# Make sure to register your dataset's metadata beforehand
# Here it's assumed you have a metadata variable set up for the validation dataset
dataset_dicts = DatasetCatalog.get("my_dataset_val")
metadata = MetadataCatalog.get("my_dataset_val")
    
# Assuming 'outputs' is the output of the predictor
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW)   # for black-white image
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    class_name = get_class_name_from_filename(d["file_name"])  # your function to get class name
    plt.figure(figsize=(14, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.text(10, 20, class_name, fontsize=14, color='white')  # Adjust position, fontsize, and color as needed
    plt.show()

