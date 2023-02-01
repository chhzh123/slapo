# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import nn


class Pattern(nn.Module):
    def forward(self, *args):
        raise NotImplementedError


def call_pattern(mod_name, *args):
    raise NotImplementedError
