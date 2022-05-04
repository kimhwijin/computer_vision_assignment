from enum import Enum, auto, unique

@unique
class MASK_STYLE(Enum):
    MULTI_CLASS_ONE_LABEL = auto()
    MULTI_CLASS_MULTI_LABEL = auto()