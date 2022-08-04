# pre-download the weights for 256 resolution model to checkpoints/ffhq256_autoenc and checkpoints/ffhq256_autoenc_cls
# wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# bunzip2 shape_predictor_68_face_landmarks.dat.bz2

import os
import torch
from torchvision.utils import save_image
import tempfile
from templates import *
from templates_cls import *
from experiment_classifier import ClsModel
from align import LandmarksDetector, image_align
from cog import BasePredictor, Path, Input, BaseModel


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):
        self.aligned_dir = "aligned"
        os.makedirs(self.aligned_dir, exist_ok=True)
        self.device = "cuda:0"

        # Model Initialization
        model_config = ffhq256_autoenc()
        self.model = LitModel(model_config)
        state = torch.load("checkpoints/ffhq256_autoenc/last.ckpt", map_location="cpu")
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.ema_model.eval()
        self.model.ema_model.to(self.device)

        # Classifier Initialization
        classifier_config = ffhq256_autoenc_cls()
        classifier_config.pretrain = None  # a bit faster
        self.classifier = ClsModel(classifier_config)
        state_class = torch.load(
            "checkpoints/ffhq256_autoenc_cls/last.ckpt", map_location="cpu"
        )
        print("latent step:", state_class["global_step"])
        self.classifier.load_state_dict(state_class["state_dict"], strict=False)
        self.classifier.to(self.device)

        self.landmarks_detector = LandmarksDetector(
            "shape_predictor_68_face_landmarks.dat"
        )

    def predict(
        self,
        image: Path = Input(
            description="Input image for face manipulation. Image will be aligned and cropped, "
            "output aligned and manipulated images.",
        ),
        target_class: str = Input(
            default="Bangs",
            choices=[
                "5_o_Clock_Shadow",
                "Arched_Eyebrows",
                "Attractive",
                "Bags_Under_Eyes",
                "Bald",
                "Bangs",
                "Big_Lips",
                "Big_Nose",
                "Black_Hair",
                "Blond_Hair",
                "Blurry",
                "Brown_Hair",
                "Bushy_Eyebrows",
                "Chubby",
                "Double_Chin",
                "Eyeglasses",
                "Goatee",
                "Gray_Hair",
                "Heavy_Makeup",
                "High_Cheekbones",
                "Male",
                "Mouth_Slightly_Open",
                "Mustache",
                "Narrow_Eyes",
                "Beard",
                "Oval_Face",
                "Pale_Skin",
                "Pointy_Nose",
                "Receding_Hairline",
                "Rosy_Cheeks",
                "Sideburns",
                "Smiling",
                "Straight_Hair",
                "Wavy_Hair",
                "Wearing_Earrings",
                "Wearing_Hat",
                "Wearing_Lipstick",
                "Wearing_Necklace",
                "Wearing_Necktie",
                "Young",
            ],
            description="Choose manipulation direction.",
        ),
        manipulation_amplitude: float = Input(
            default=0.3,
            ge=-0.5,
            le=0.5,
            description="When set too strong it would result in artifact as it could dominate the original image information.",
        ),
        T_step: int = Input(
            default=100,
            choices=[50, 100, 125, 200, 250, 500],
            description="Number of step for generation.",
        ),
        T_inv: int = Input(default=200, choices=[50, 100, 125, 200, 250, 500]),
    ) -> List[ModelOutput]:

        img_size = 256
        print("Aligning image...")
        for i, face_landmarks in enumerate(
            self.landmarks_detector.get_landmarks(str(image)), start=1
        ):
            image_align(str(image), f"{self.aligned_dir}/aligned.png", face_landmarks)

        data = ImageDataset(
            self.aligned_dir,
            image_size=img_size,
            exts=["jpg", "jpeg", "JPG", "png"],
            do_augment=False,
        )

        print("Encoding and Manipulating the aligned image...")
        cls_manipulation_amplitude = manipulation_amplitude
        interpreted_target_class = target_class
        if (
            target_class not in CelebAttrDataset.id_to_cls
            and f"No_{target_class}" in CelebAttrDataset.id_to_cls
        ):
            cls_manipulation_amplitude = -manipulation_amplitude
            interpreted_target_class = f"No_{target_class}"

        batch = data[0]["img"][None]

        semantic_latent = self.model.encode(batch.to(self.device))
        stochastic_latent = self.model.encode_stochastic(
            batch.to(self.device), semantic_latent, T=T_inv
        )

        cls_id = CelebAttrDataset.cls_to_id[interpreted_target_class]
        class_direction = self.classifier.classifier.weight[cls_id]
        normalized_class_direction = F.normalize(class_direction[None, :], dim=1)

        normalized_semantic_latent = self.classifier.normalize(semantic_latent)
        normalized_manipulation_amp = cls_manipulation_amplitude * math.sqrt(512)
        normalized_manipulated_semantic_latent = (
            normalized_semantic_latent
            + normalized_manipulation_amp * normalized_class_direction
        )

        manipulated_semantic_latent = self.classifier.denormalize(
            normalized_manipulated_semantic_latent
        )

        # Render Manipulated image
        manipulated_img = self.model.render(
            stochastic_latent, manipulated_semantic_latent, T=T_step
        )[0]
        original_img = data[0]["img"]

        model_output = []
        out_path = Path(tempfile.mkdtemp()) / "original_aligned.png"
        save_image(convert2rgb(original_img), str(out_path))
        model_output.append(ModelOutput(image=out_path))

        out_path = Path(tempfile.mkdtemp()) / "manipulated_img.png"
        save_image(convert2rgb(manipulated_img, adjust_scale=False), str(out_path))
        model_output.append(ModelOutput(image=out_path))
        return model_output


def convert2rgb(img, adjust_scale=True):
    convert_img = torch.tensor(img)
    if adjust_scale:
        convert_img = (convert_img + 1) / 2
    return convert_img.cpu()
