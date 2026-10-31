# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

import torchopt
from torchopt import pytree
from torchopt.alias.utils import _get_use_chain_flat, _set_use_chain_flat


if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(params=[True, False], ids=['flat', 'pytree'])
def use_chain_flat(request: pytest.FixtureRequest) -> Generator[bool, None, None]:
    previous = _get_use_chain_flat()
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    _set_use_chain_flat(request.param)
    try:
        yield request.param
    finally:
        _set_use_chain_flat(previous)
        torch.set_default_dtype(previous_dtype)


@pytest.mark.parametrize('mask_values', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('callable_mask', [True, False])
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('use_accelerated_op', [True, False])
def test_masked_adamw_matches_parameter_groups(
    use_chain_flat: bool,
    mask_values: tuple[bool, bool],
    callable_mask: bool,
    inplace: bool,
    use_accelerated_op: bool,
) -> None:
    params = [
        torch.tensor([[0.7, -1.2]], dtype=torch.float64),
        torch.tensor([0.4, -0.6], dtype=torch.float64),
        torch.tensor([2.3], dtype=torch.float64),
    ]
    reference = [param.clone() for param in params]
    mask = [mask_values[0], mask_values[1], mask_values[1]]
    if callable_mask:

        def mask_fn(tree):
            return pytree.tree_map(lambda p: mask_values[0 if p.ndim == 2 else 1], tree)

        mask = mask_fn
    optimizer = torchopt.adamw(
        lr=0.2,
        betas=(0.8, 0.9),
        weight_decay=0.1,
        mask=mask,
        use_accelerated_op=use_accelerated_op,
    )
    state = optimizer.init(params)
    reference_optimizer = torch.optim.AdamW(
        [
            {'params': [reference[0]], 'weight_decay': 0.1 if mask_values[0] else 0.0},
            {'params': reference[1:], 'weight_decay': 0.1 if mask_values[1] else 0.0},
        ],
        lr=0.2,
        betas=(0.8, 0.9),
    )
    for step in range(4):
        grads = [2 * param + step / 10 for param in params[:2]] + [None]
        for param, grad in zip(reference, grads):
            param.grad = None if grad is None else grad.clone()
        updates, state = optimizer.update(grads, state, params=params, inplace=inplace)
        assert updates[2] is None
        params = torchopt.apply_updates(params, updates, inplace=inplace)
        reference_optimizer.step()
        for actual, expected in zip(params, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_masked_adamw_meta_gradient(use_chain_flat: bool) -> None:
    initial = [
        torch.tensor([[0.7, -1.2]], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.4, -0.6], dtype=torch.float64, requires_grad=True),
    ]
    reference_initial = [param.detach().clone().requires_grad_() for param in initial]
    params = initial
    reference = reference_initial
    optimizer = torchopt.adamw(
        lr=0.2,
        betas=(0.8, 0.9),
        eps=1e-4,
        eps_root=1e-3,
        weight_decay=0.1,
        mask=lambda tree: pytree.tree_map(lambda p: p.ndim != 1, tree),
    )
    state = optimizer.init(params)
    first_moments = [torch.zeros_like(param) for param in reference]
    second_moments = [torch.zeros_like(param) for param in reference]
    for step in range(1, 4):
        loss = (params[0] + params[1]).square().sum()
        grads = list(torch.autograd.grad(loss, params, create_graph=True))
        updates, state = optimizer.update(grads, state, params=params, inplace=False)
        params = torchopt.apply_updates(params, updates, inplace=False)

        reference_loss = (reference[0] + reference[1]).square().sum()
        reference_grads = torch.autograd.grad(reference_loss, reference, create_graph=True)
        next_reference = []
        for index, (param, grad) in enumerate(zip(reference, reference_grads)):
            first_moments[index] = 0.8 * first_moments[index] + 0.2 * grad
            second_moments[index] = 0.9 * second_moments[index] + 0.1 * grad.square()
            mean = first_moments[index] / (1 - 0.8**step)
            variance = second_moments[index] / (1 - 0.9**step)
            update = mean / ((variance + 1e-3).sqrt() + 1e-4)
            if index == 0:
                update = update + 0.1 * param
            next_reference.append(param - 0.2 * update)
        reference = next_reference
        for actual, expected in zip(params, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    outer_loss = sum(param.square().sum() for param in params)
    reference_outer_loss = sum(param.square().sum() for param in reference)
    meta_grads = torch.autograd.grad(outer_loss, initial)
    reference_meta_grads = torch.autograd.grad(reference_outer_loss, reference_initial)
    for actual, expected in zip(meta_grads, reference_meta_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('inplace', [True, False])
def test_masked_adam_state(use_chain_flat: bool, inplace: bool) -> None:
    params = [torch.tensor([2.0]), torch.tensor([3.0])]
    inner = torchopt.transform.scale_by_adam.impl(already_flattened=use_chain_flat)
    optimizer = torchopt.transform.masked.impl(
        inner, [True, False], already_flattened=use_chain_flat,
    )
    state = optimizer.init(params)
    # Adam stores two moments and a counter for only the selected parameter.
    assert len(pytree.tree_leaves(state)) == 3
    for _ in range(3):
        grads = [torch.tensor([2.0]), torch.tensor([3.0])]
        passed_through = grads[1]
        updates, state = optimizer.update(grads, state, inplace=inplace)
        assert updates[1] is passed_through
        torch.testing.assert_close(updates[0], torch.tensor([2.0 / (2.0 + 1e-8)]))
        assert len(pytree.tree_leaves(state)) == 3
