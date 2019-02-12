"""Santa for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.training import slot_creator
from tensorflow.python.training.checkpointable import base as checkpointable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np


class _RefVariableProcessor(optimizer._OptimizableVariable):
    """Processor for Variable."""

    def __init__(self, var):
        self.var = var

    def __str__(self):
        return "<_RefVariableProcessor(%s)>" % self.var

    def target(self):
        return self.var._ref()

    def update_op(self, optimizer, grad, global_step):
        if isinstance(grad, ops.Tensor):
            update_op = optimizer._apply_dense(grad, self.var, global_step)
            if self.var.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return self.var.assign(self.var.constraint(self.var))
            else:
                return update_op
        else:
            assert isinstance(grad, ops.IndexedSlices), ("Gradient ", grad, " is neither a tensor nor IndexedSlices.")
            if self.var.constraint is not None:
                raise RuntimeError("Cannot use a constraint function on a sparse variable.")
            return optimizer._apply_sparse_duplicate_indices(grad, self.var)


def _get_processor(var):
    """The processor of var."""
    if context.executing_eagerly():
        if isinstance(var, ops.Tensor):
            return optimizer._TensorProcessor(var)
        else:
            return optimizer._DenseResourceVariableProcessor(var)
    if isinstance(var, resource_variable_ops.ResourceVariable) and not var._in_graph_mode:
        # True if and only if `var` was initialized eagerly.
        return optimizer._DenseResourceVariableProcessor(var)
    if var.op.type == "VarHandleOp":
        return optimizer._DenseResourceVariableProcessor(var)
    if isinstance(var, variables.Variable):
        return _RefVariableProcessor(var)
    if isinstance(var, ops.Tensor):
        return optimizer._TensorProcessor(var)
    raise NotImplementedError("Trying to optimize unsupported type ", var)


class SantaOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Santa algorithm.
    See [Chen et al., 2016](https://arxiv.org/pdf/1512.07962.pdf)
    """

    def __init__(self, eta=1e-6, gamma=0.5, sigma=0.999, epsilon=1e-8,
                 burnin=10000, const=1000, use_locking=False, name="Santa"):
        """Construct a new Santa optimizer.
        Args:
          alpha1: A Tensor or a floating point value.  
            The learning rate.
          beta1: A float value or a constant float tensor.
            The exponential decay rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor.
            The exponential decay rate for the 2nd moment estimates.
          beta3: A float value or a constant float tensor.
            The exponential decay rate for computing relative change.
          epsilon: A float value or a constant float tensor.
            A small constant for numerical stability. 
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Santa".
        @compatibility(eager)
        When eager execution is enabled, `eta`, `gamma`, `sigma`, `epsilon`, `burnin`, 
        and `const` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """
        super(SantaOptimizer, self).__init__(use_locking, name)
        self.eta = eta
        self.gamma = gamma
        self.sigma = sigma
        self.epsilon = epsilon
        self.burnin = burnin
        self.const = const

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for var in var_list:
            self._get_or_make_slot_with_initializer(
                var, tf.zeros_initializer(),
                var.shape, var.dtype, "v", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.constant_initializer(1 / np.sqrt(self.epsilon)),
                var.shape, var.dtype, "g", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.constant_initializer(np.sqrt(self.eta) * self.const),
                var.shape, var.dtype, "a", self._name
            )
            self._get_or_make_slot_with_initializer(
                var, tf.random_normal_initializer(stddev=np.sqrt(self.eta)),
                var.shape, var.dtype, "u", self._name
            )

    def _prepare(self):
        self.eta = tf.convert_to_tensor(
            value=self._call_if_callable(self.eta),
            name="eta"
        )
        self.gamma = tf.convert_to_tensor(
            value=self._call_if_callable(self.gamma),
            name="gamma"
        )
        self.sigma = tf.convert_to_tensor(
            value=self._call_if_callable(self.sigma),
            name="sigma"
        )
        self.epsilon = tf.convert_to_tensor(
            value=self._call_if_callable(self.epsilon),
            name="epsilon"
        )
        self.burnin = tf.convert_to_tensor(
            value=self._call_if_callable(self.burnin),
            name="burnin"
        )

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.
        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.
        Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
        global_step: Optional `Variable` to increment by one after the
            variables have been updated.
        name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
        Returns:
        An `Operation` that applies the specified gradients. If `global_step`
        was not None, that operation also increments `global_step`.
        Raises:
        TypeError: If `grads_and_vars` is malformed.
        ValueError: If none of the variables have gradients.
        RuntimeError: If you should use `_distributed_apply()` instead.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        # Handle DistributionStrategy case.
        if distribution_strategy_context.get_cross_tower_context():
            raise RuntimeError("Use `_distributed_apply()` instead of `apply_gradients()` in a cross-tower context.")
        # TODO(isaprykin): Get rid of `has_distribution_strategy()` check by
        # always calling _distributed_apply(), using the default distribution
        # as needed.
        if distribution_strategy_context.has_distribution_strategy():
            grads_and_vars = optimizer.get_filtered_grad_fn(lambda: grads_and_vars)()
            return distribution_strategy_context.get_tower_context().merge_call(
                self._distributed_apply, grads_and_vars, global_step, name
            )

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    grad = ops.convert_to_tensor_or_indexed_slices(grad)
                except TypeError:
                    raise TypeError("Gradient must be convertible to a Tensor or IndexedSlices, or None: %s" % grad)
                if not isinstance(grad, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError("Gradient must be a Tensor, IndexedSlices, or None: %s" % grad)
            processor = _get_processor(var)
            converted_grads_and_vars.append((grad, var, processor))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [var for grad, var, _ in converted_grads_and_vars if grad is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." % ([str(var) for _, var, _ in converted_grads_and_vars],))
        with ops.init_scope():
            self._create_slots(var_list)
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                # TODO(apassos): figure out how to get the variable name here.
                if context.executing_eagerly() or isinstance(var, resource_variable_ops.ResourceVariable) and not var._in_graph_mode:
                    scope_name = ""
                else:
                    scope_name = var.op.name
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, grad, global_step))
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        if isinstance(global_step, resource_variable_ops.ResourceVariable):
                            # TODO(apassos): the implicit read in assign_add is slow; consider
                            # making it less so.
                            apply_updates = resource_variable_ops.assign_add_variable_op(
                                resource=global_step.handle,
                                value=ops.convert_to_tensor(
                                    value=1,
                                    dtype=global_step.dtype
                                ),
                                name=name
                            )
                        else:
                            apply_updates = state_ops.assign_add(
                                ref=global_step,
                                value=1,
                                name=name
                            )

            if not context.executing_eagerly():
                if isinstance(apply_updates, ops.Tensor):
                    apply_updates = apply_updates.op
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if apply_updates not in train_op:
                    train_op.append(apply_updates)

            return apply_updates

    def _apply_dense(self, grad, var, global_step):

        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        a = self.get_slot(var, "a")
        u = self.get_slot(var, "u")

        eta = math_ops.cast(self.eta, var.dtype.base_dtype)
        gamma = math_ops.cast(self.gamma, var.dtype.base_dtype)
        sigma = math_ops.cast(self.sigma, var.dtype.base_dtype)
        epsilon = math_ops.cast(self.epsilon, var.dtype.base_dtype)
        burnin = math_ops.cast(self.burnin, global_step.dtype.base_dtype)

        def update(exploration):

            beta = tf.cast(global_step + 1, var.dtype) ** gamma
            zeta = tf.random_normal(var.shape)

            v_ = sigma * v + (1 - sigma) * grad * grad
            g_ = 1 / (epsilon + v_ ** 0.5) ** 0.5

            var_ = var + g_ * u / 2

            if exploration:
                a_ = a + (u * u - eta / beta) / 2
                u_ = tf.exp(- a_ / 2) * u
                u_ = u_ - eta * g_ * grad
                # u_ = u_ + (2 * eta / beta * g) ** 0.5 * zeta
                # u_ = u_ + eta / beta * (1 - g / g_) / u
                u_ = u_ + (2 * eta ** 1.5 / beta * g) ** 0.5 * zeta
                u_ = tf.exp(- a_ / 2) * u_
                a_ = a_ + (u_ * u_ - eta / beta) / 2
            else:
                a_ = a
                u_ = tf.exp(- a_ / 2) * u
                u_ = u_ - eta * g_ * grad
                u_ = tf.exp(- a_ / 2) * u_

            var_ = var_ + g_ * u_ / 2

            return var_, v_, g_, a_, u_

        var_, v_, g_, a_, u_ = tf.cond(
            pred=tf.less(global_step, burnin),
            true_fn=lambda: update(True),
            false_fn=lambda: update(False)
        )

        return tf.group(*[
            var.assign(var_),
            v.assign(v_),
            g.assign(g_),
            a.assign(a_),
            u.assign(u_),
        ])
