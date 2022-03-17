from enum import Enum
from torch import nn


class TrainMode(Enum):
    # manipulate mode = training the classifier
    manipulate = 'manipulate'
    # default trainin mode!
    diffusion = 'diffusion'
    # default latent training mode!
    # fitting the a DDPM to a given latent
    latent_diffusion = 'latentdiffusion'

    def is_manipulate(self):
        return self in [
            TrainMode.manipulate,
        ]

    def is_diffusion(self):
        return self in [
            TrainMode.diffusion,
            TrainMode.latent_diffusion,
        ]

    def is_autoenc(self):
        # the network possibly does autoencoding
        return self in [
            TrainMode.diffusion,
        ]

    def is_latent_diffusion(self):
        return self in [
            TrainMode.latent_diffusion,
        ]

    def use_latent_net(self):
        return self.is_latent_diffusion()

    def require_dataset_infer(self):
        """
        whether training in this mode requires the latent variables to be available?
        """
        # this will precalculate all the latents before hand
        # and the dataset will be all the predicted latents
        return self in [
            TrainMode.latent_diffusion,
            TrainMode.manipulate,
        ]


class ManipulateMode(Enum):
    """
    how to train the classifier to manipulate
    """
    # train on whole celeba attr dataset
    celebahq_all = 'celebahq_all'
    # celeba with D2C's crop
    d2c_fewshot = 'd2cfewshot'
    d2c_fewshot_allneg = 'd2cfewshotallneg'

    def is_celeba_attr(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
            ManipulateMode.celebahq_all,
        ]

    def is_single_class(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot(self):
        return self in [
            ManipulateMode.d2c_fewshot,
            ManipulateMode.d2c_fewshot_allneg,
        ]

    def is_fewshot_allneg(self):
        return self in [
            ManipulateMode.d2c_fewshot_allneg,
        ]


class ModelType(Enum):
    """
    Kinds of the backbone models
    """

    # unconditional ddpm
    ddpm = 'ddpm'
    # autoencoding ddpm cannot do unconditional generation
    autoencoder = 'autoencoder'

    def has_autoenc(self):
        return self in [
            ModelType.autoencoder,
        ]

    def can_sample(self):
        return self in [ModelType.ddpm]


class ChamferType(Enum):
    chamfer = 'chamfer'
    stochastic = 'stochastic'


class ModelName(Enum):
    """
    List of all supported model classes
    """

    beatgans_ddpm = 'beatgans_ddpm'
    beatgans_autoenc = 'beatgans_autoenc'


class EncoderName(Enum):
    """
    List of all encoders for ddpm models
    """

    v1 = 'v1'
    v2 = 'v2'


class ModelMeanType(Enum):
    """
    Which type of output the model predicts.
    """

    prev_x = 'x_prev'  # the model predicts x_{t-1}
    start_x = 'x_start'  # the model predicts x_0
    eps = 'eps'  # the model predicts epsilon
    scaled_start_x = 'scaledxstart'  # the model predicts sqrt(alphacum) x_0


class ModelVarType(Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    # learned directly
    learned = 'learned'
    # posterior beta_t
    fixed_small = 'fixed_small'
    # beta_t
    fixed_large = 'fixed_large'
    # predict values between FIXED_SMALL and FIXED_LARGE, making its job easier
    learned_range = 'learned_range'


class LossType(Enum):
    mse = 'mse'  # use raw MSE loss (and KL when learning variances)
    l1 = 'l1'
    # mse weighted by the variance, somewhat like in kl
    mse_var_weighted = 'mse_weighted'
    mse_rescaled = 'mse_rescaled'  # use raw MSE loss (with RESCALED_KL when learning variances)
    kl = 'kl'  # use the variational lower-bound
    kl_rescaled = 'kl_rescaled'  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.kl or self == LossType.kl_rescaled


class MSEWeightType(Enum):
    # use the ddpm's default variance (either analytical or learned)
    var = 'var'
    # optimal variance by deriving the min kl per image (based on mse of epsilon)
    # = small sigma + mse
    var_min_kl_img = 'varoptimg'
    # optimal variance regradless of the posterior sigmas
    # = mse only
    var_min_kl_mse_img = 'varoptmseimg'
    # same as the above but is based on mse of mu of xprev
    var_min_kl_xprev_img = 'varoptxprevimg'


class XStartWeightType(Enum):
    # weights for the mse of the xstart
    # unweighted x start
    uniform = 'uniform'
    # reciprocal 1 - alpha_bar
    reciprocal_alphabar = 'recipalpha'
    # same as the above but not exceeding mse = 1
    reciprocal_alphabar_safe = 'recipalphasafe'
    # turning x0 into eps as use the mse(eps)
    eps = 'eps'
    # the same as above but not turning into eps
    eps2 = 'eps2'
    # same as the above but not exceeding mse = 1
    eps2_safe = 'eps2safe'
    eps_huber = 'epshuber'
    unit_mse_x0 = 'unitmsex0'
    unit_mse_eps = 'unitmseeps'


class GenerativeType(Enum):
    """
    How's a sample generated
    """

    ddpm = 'ddpm'
    ddim = 'ddim'


class OptimizerType(Enum):
    adam = 'adam'
    adamw = 'adamw'


class Activation(Enum):
    none = 'none'
    relu = 'relu'
    lrelu = 'lrelu'
    silu = 'silu'
    tanh = 'tanh'

    def get_act(self):
        if self == Activation.none:
            return nn.Identity()
        elif self == Activation.relu:
            return nn.ReLU()
        elif self == Activation.lrelu:
            return nn.LeakyReLU(negative_slope=0.2)
        elif self == Activation.silu:
            return nn.SiLU()
        elif self == Activation.tanh:
            return nn.Tanh()
        else:
            raise NotImplementedError()


class ManipulateLossType(Enum):
    bce = 'bce'
    mse = 'mse'