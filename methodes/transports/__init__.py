from methodes.transports.edm import EDM
from methodes.transports.linear import Linear
from methodes.transports.random import Random
from methodes.transports.relinear import ReLinear
from methodes.transports.trigflow import TrigFlow
from methodes.transports.triglinear import TrigLinear


TRANSPORTS = {
    "EDM": EDM,
    "Linear": Linear,
    "Random": Random,
    "ReLinear": ReLinear,
    "TrigFlow": TrigFlow,
    "TrigLinear": TrigLinear,
}
