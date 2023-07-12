# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Annotation primitive."""
# pylint: disable=arguments-differ

from .base import register_primitive, Primitive


@register_primitive()
class CudaGraphifyPrimitive(Primitive):

    @staticmethod
    def name():
        return "cudagraphify"

    @staticmethod
    def apply(sch, example_inputs):
        if not sch.get_top_schedule().metadata.primitives["cudagraphify"]:
            sch.get_top_schedule().metadata.primitives["cudagraphify"] = [
                (sch, example_inputs)
            ]
        else:
            sch.get_top_schedule().metadata.primitives["cudagraphify"].append(
                (sch, example_inputs)
            )
