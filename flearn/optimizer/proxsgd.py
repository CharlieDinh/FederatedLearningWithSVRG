from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import flearn.utils.tf_utils as tf_utils


class PROXSGD(optimizer.Optimizer):
    """Implementation of Proximal Gradient Decent, i.e., FedProx optimizer"""

    def __init__(self, learning_rate=0.001, lamb=0.001, use_locking=False, name="PROXSGD"):
        super(PROXSGD, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lamb = lamb
        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._lamb_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._lamb_t = ops.convert_to_tensor(self._lamb, name="lamb")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        lamb_t = math_ops.cast(self._lamb_t, var.dtype.base_dtype)

        prox = prox = tf_utils.prox_L2(var - lr_t*grad, lamb_t)
        var_update = state_ops.assign(var, prox)

        return control_flow_ops.group(*[var_update, ])

