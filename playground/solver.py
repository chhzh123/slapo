# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp
import z3

# number of devices
p = 2


class ShardSpec:
    def __init__(self, spec):
        self.map = {"RR": 0, "RS": 1, "SR": 2}
        if isinstance(spec, str):
            self.spec = spec
        else:
            self.spec = list(self.map.keys())[list(self.map.values()).index(spec)]

    @property
    def id(self):
        return self.map[self.spec]

    def __str__(self):
        return self.spec


reshard_cost_map = {
    "RR": {"RR": 0, "RS": 0, "SR": 0},
    "RS": {"RR": 1 / p, "RS": 0, "SR": 1 / p - 1 / (p * p)},
    "SR": {"RR": 1 / p, "RS": 1 / p - 1 / (p * p), "SR": 0},
}


def calculate_reshard_cost(prev, curr, shape):
    return int(reshard_cost_map[ShardSpec(prev).spec][ShardSpec(curr).spec] * shape)


def calculate_reshard_cost_z3(prev, curr, shape):
    result = 1e12  # invalid
    for in_spec, target_map in reshard_cost_map.items():
        tmp = 1e12  # invalid
        for out_spec, val in target_map.items():
            tmp = z3.If(curr == ShardSpec(out_spec).id, int(val * shape), tmp)
        result = z3.If(prev == ShardSpec(in_spec).id, tmp, result)
    return result


class MatmulOp:
    def __init__(self, name, lhs_shape, rhs_shape, out_shape):
        self.name = name
        self.lhs_shape = lhs_shape
        self.rhs_shape = rhs_shape
        self.out_shape = out_shape
        self.lhs_size = lhs_shape[-2] * lhs_shape[-1]
        assert lhs_shape[-1] == rhs_shape[-1]
        # weight is transposed
        self.rhs_size = rhs_shape[-1] * rhs_shape[-2]
        self.out_size = lhs_shape[-2] * rhs_shape[-2]
        self.output_map = {"RR": "RS", "RS": "RR", "SR": "SR"}
        self.comm_cost_map = {  # map from input spec to comm cost
            "RR": 0,
            "RS": self.out_size,  # all_reduce
            "SR": 0,
        }

    def generate_output(self, lhs, rhs):
        return ShardSpec(self.output_map[ShardSpec(lhs).spec]).id

    def generate_output_z3(self, lhs, rhs):
        result = 3  # invalid
        for inp, out in self.output_map.items():
            result = z3.If(lhs == ShardSpec(inp).id, ShardSpec(out).id, result)
        return result

    def calculate_comm_cost(self, lhs, rhs):
        return self.comm_cost_map[ShardSpec(lhs).spec]

    def calculate_comm_cost_z3(self, lhs, rhs):
        result = 1e12  # invalid
        for inp, cost in self.comm_cost_map.items():
            result = z3.If(lhs == ShardSpec(inp).id, cost, result)
        return result


class Solver:
    def __init__(self, gm) -> None:
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())
        self.z3_graph = []
        self.goal = []
        self.cost = None
        self.shard_spec = {}  # {node_name: shard_spec}

    def inference_shape(self, inputs):
        sp = ShapeProp(self.gm)
        sp.propagate(*inputs)
        for node in self.gm.graph.nodes:
            if "tensor_meta" in node.meta:
                if isinstance(node.meta["tensor_meta"], list):
                    lst = node.meta["tensor_meta"]
                else:
                    lst = [node.meta["tensor_meta"]]
                for data in lst:
                    print(node.name, data)

    def construct_z3_problem(self):
        bitvecs = {}
        input_constraints = []
        format_constraints = []
        outputs = []
        comm_costs = []
        for op in self.z3_graph:
            # no need to include output, since output can be obtained from inputs
            name = op.name
            lhs = z3.BitVec(f"{name}_lhs", 2)  # input
            rhs = z3.BitVec(f"{name}_rhs", 2)  # weight
            bitvecs[f"{name}_lhs"] = lhs
            bitvecs[f"{name}_rhs"] = rhs

            # input constraints
            constraints = [
                z3.And(lhs == ShardSpec("RR").id, rhs == ShardSpec("RS").id),
                z3.And(lhs == ShardSpec("RS").id, rhs == ShardSpec("SR").id),
                z3.And(lhs == ShardSpec("SR").id, rhs == ShardSpec("RR").id),
            ]
            input_constraints.append(z3.Or(*constraints))

            # format constraints
            format_constraints.extend([z3.ULE(lhs, 3), z3.ULE(rhs, 3)])

            # output
            outputs.append(op.generate_output_z3(lhs, rhs))

            # communication cost
            comm_costs.append(op.calculate_comm_cost_z3(lhs, rhs))

        reshard_costs = []
        for i, op in enumerate(self.z3_graph):
            prev = outputs[i]
            curr = (
                bitvecs[f"{self.z3_graph[i + 1].name}_lhs"]
                if i < len(self.z3_graph) - 1
                else ShardSpec("RR").id
            )
            reshard_costs.append(calculate_reshard_cost_z3(prev, curr, op.out_size))

        self.cost = sum(comm_costs) + sum(reshard_costs)

        self.goal += [
            bitvecs[f"{self.z3_graph[0].name}_lhs"] == ShardSpec("RR").id
        ]  # input should not be sharded
        self.goal += input_constraints
        self.goal += format_constraints

    def solve(self, inputs):
        self.inference_shape(inputs)
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":  # input
                continue
            elif node.op == "call_module":
                mod = self.named_modules[node.target]
                if isinstance(mod, nn.Linear):
                    self.z3_graph.append(
                        MatmulOp(
                            node.name,
                            node.args[0].meta["tensor_meta"].shape,
                            mod.weight.shape,
                            node.meta["tensor_meta"].shape,
                        )
                    )
            elif node.op == "call_function":
                if node.target == torch.matmul:
                    self.z3_graph.append(
                        MatmulOp(
                            node.name,
                            node.args[0].meta["tensor_meta"].shape,
                            node.args[1].meta["tensor_meta"].shape,
                            node.meta["tensor_meta"].shape,
                        )
                    )
        self.construct_z3_problem()
        sol = z3.Solver()
        max_cost = 1e12
        for it in range(3):
            print(f"=================== Iter {it} ===================")
            sol.add(self.goal)
            sol.push()
            assert self.cost is not None
            sol.add(self.cost < max_cost)
            # print(sol)
            sat = sol.check()
            if str(sat) == "unsat":
                print("Cannot find solution")
                break
            mod = sol.model()
            print(mod)
            results = {d.name(): mod[d] for d in mod.decls()}
            max_cost = 0
            for i, op in enumerate(self.z3_graph):
                name = op.name
                lhs = results[f"{name}_lhs"]
                rhs = results[f"{name}_rhs"]
                print(name, op.lhs_shape, op.rhs_shape, op.out_shape)
                output = op.generate_output(lhs, rhs)
                print(
                    f"{name}: {ShardSpec(lhs)} x {ShardSpec(rhs)} = {ShardSpec(output)}"
                )
                comm_cost = op.calculate_comm_cost(lhs, rhs)
                next_inp = (
                    results[f"{self.z3_graph[i + 1].name}_lhs"]
                    if i < len(self.z3_graph) - 1
                    else ShardSpec("RR").id
                )
                reshard_cost = calculate_reshard_cost(output, next_inp, op.out_size)
                max_cost += comm_cost + reshard_cost
                print(comm_cost, reshard_cost)
            print("Total cost:", max_cost)
            sol.pop()
