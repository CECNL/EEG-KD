from .sccnet import SCCNet22, SCCNet7, SCCNet4
from .eegnet import EEGNet22, EEGNet7, EEGNet4
from .shallow import ShallowConvNet22, ShallowConvNet7, ShallowConvNet4


model_dict = {
    'SCCNet22': SCCNet22,
    'SCCNet7' : SCCNet7,
    'SCCNet4' : SCCNet4,
    'EEGNet22': EEGNet22,
    'EEGNet7' : EEGNet7,
    'EEGNet4' : EEGNet4,
    'Shallow22': ShallowConvNet22,
    'Shallow7' : ShallowConvNet7,
    'Shallow4' : ShallowConvNet4,
}