from enum import Enum, auto, unique

@unique
class MASK_STYLE(Enum):
    MULTI_CLASS_MULTI_LABEL = auto()

@unique
class KERNEL_INITIALIZER(Enum):
    HE_UNIFORM = 'he_uniform'
    HE_NORMAL = 'he_normal'
    GLOROT_UNIFORM = 'glorot_uniform'
    GLOROT_NORMAL = 'glorot_normal'

@unique
class ACTIVATION(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'

