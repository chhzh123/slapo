# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp
import z3


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


class FxOp:
    def __init__(self, node):
        self.node = node
        self.name = node.name
        self.args = []
        self.users = []
        self.out_shape = node.meta["tensor_meta"].shape
        self.out_size = self.out_shape[-2] * self.out_shape[-1]

    def add_arg(self, arg):
        self.args.append(arg)

    def add_user(self, user):
        self.users.append(user)


class PlaceholderOp(FxOp):
    def generate_input_z3(self):
        self.x = z3.BitVec(f"{self.name}", 2)
        # input should not be sharded
        return [self.x], [self.x == ShardSpec("RR").id]

    def generate_output(self):
        return ShardSpec("RR").id

    def generate_output_z3(self):
        return self.x

    def calculate_comm_cost(self):
        return 0

    def calculate_comm_cost_z3(self):
        return 0


class ElementwiseOp(FxOp):
    def generate_input_z3(self):
        return [], []

    def generate_output(self):
        return self.args[0].generate_output()

    def generate_output_z3(self):
        return self.args[0].generate_output_z3()

    def calculate_comm_cost_z3(self):
        return 0


class MatmulOp(FxOp):
    def __init__(self, node, mod=None, is_linear=False):
        super().__init__(node)
        self.lhs_shape = node.args[0].meta["tensor_meta"].shape
        self.rhs_shape = (
            node.args[1].meta["tensor_meta"].shape
            if not is_linear
            else mod.weight.shape
        )
        self.out_shape = (
            node.meta["tensor_meta"].shape
            if not isinstance(node.meta["tensor_meta"], list)
            else node.meta["tensor_meta"][0].shape
        )
        self.lhs_size = self.lhs_shape[-2] * self.lhs_shape[-1]
        print(self.name, self.lhs_shape, self.rhs_shape, self.out_shape)
        if is_linear:
            # weight is transposed
            assert self.lhs_shape[-1] == self.rhs_shape[-1]
            self.rhs_size = self.rhs_shape[-1] * self.rhs_shape[-2]
            self.out_size = self.lhs_shape[-2] * self.rhs_shape[-2]
        else:
            assert self.lhs_shape[-1] == self.rhs_shape[-2]
            self.rhs_size = self.rhs_shape[-2] * self.rhs_shape[-1]
            self.out_size = self.lhs_shape[-2] * self.rhs_shape[-1]
        self.output_map = {"RR": "RS", "RS": "RR", "SR": "SR"}
        self.comm_cost_map = {  # map from input spec to comm cost
            "RR": 0,
            "RS": self.out_size,  # all_reduce
            "SR": 0,
        }

    def generate_input_z3(self):
        self.lhs = z3.BitVec(f"{self.name}_lhs", 2)  # input
        self.rhs = z3.BitVec(f"{self.name}_rhs", 2)  # weight

        compute_constraints = [
            z3.Or(
                [
                    z3.And(
                        self.lhs == ShardSpec("RR").id, self.rhs == ShardSpec("RS").id
                    ),
                    z3.And(
                        self.lhs == ShardSpec("RS").id, self.rhs == ShardSpec("SR").id
                    ),
                    z3.And(
                        self.lhs == ShardSpec("SR").id, self.rhs == ShardSpec("RR").id
                    ),
                ]
            )
        ]
        format_constraints = [z3.ULE(self.lhs, 3), z3.ULE(self.rhs, 3)]
        constraints = compute_constraints + format_constraints
        # force to shard
        # constraints += [self.lhs != ShardSpec("RR").id, self.rhs != ShardSpec("RR").id]
        return [self.lhs, self.rhs], constraints

    def set_concrete_values(self, lhs, rhs):
        self.lhs_v = lhs
        self.rhs_v = rhs

    def generate_output(self):
        return ShardSpec(self.output_map[ShardSpec(self.lhs_v).spec]).id

    def generate_output_z3(self):
        result = 3  # invalid
        for inp, out in self.output_map.items():
            result = z3.If(self.lhs == ShardSpec(inp).id, ShardSpec(out).id, result)
        return result

    def calculate_comm_cost(self):
        return self.comm_cost_map[ShardSpec(self.lhs_v).spec]

    def calculate_comm_cost_z3(self):
        result = 1e12  # invalid
        for inp, cost in self.comm_cost_map.items():
            result = z3.If(self.lhs == ShardSpec(inp).id, cost, result)
        return result


class Solver:
    def __init__(self, gm, p) -> None:
        self.gm = gm
        self.named_modules = dict(self.gm.named_modules())
        self.z3_graph = {}  # {node_name: FxOp}
        self.goal = []
        self.cost = None
        self.num_devices = p
        self.reshard_cost_map = {
            "RR": {"RR": 0, "RS": 0, "SR": 0},
            "RS": {"RR": 1 / p, "RS": 0, "SR": 1 / p - 1 / (p * p)},
            "SR": {"RR": 1 / p, "RS": 1 / p - 1 / (p * p), "SR": 0},
        }

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

    def calculate_reshard_cost(self, prev, curr, shape):
        return int(
            self.reshard_cost_map[ShardSpec(prev).spec][ShardSpec(curr).spec] * shape
        )

    def calculate_reshard_cost_z3(self, prev, curr, shape):
        result = 1e12  # invalid
        for in_spec, target_map in self.reshard_cost_map.items():
            tmp = 1e12  # invalid
            for out_spec, val in target_map.items():
                tmp = z3.If(curr == ShardSpec(out_spec).id, int(val * shape), tmp)
            result = z3.If(prev == ShardSpec(in_spec).id, tmp, result)
        return result

    def construct_z3_graph(self):
        print(self.gm.graph)
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":  # input
                new_op = PlaceholderOp(node)
            elif node.op == "call_module":
                mod = self.named_modules[node.target]
                if isinstance(mod, nn.Linear):
                    new_op = MatmulOp(
                        node,
                        mod=mod,
                        is_linear=True,
                    )
                else:
                    raise RuntimeError(f"Unsupported module: {node.target}")
            elif node.op == "call_function":
                if node.target == torch.matmul:
                    new_op = MatmulOp(node)
                else:
                    new_op = ElementwiseOp(node)
            else:  # output
                continue
            # construct edges
            for arg in node.args:
                new_op.add_arg(self.z3_graph[arg.name])
                self.z3_graph[arg.name].add_user(new_op)
            self.z3_graph[node.name] = new_op
        print(self.z3_graph)

    def construct_z3_problem(self):
        bitvecs = {}
        input_constraints = []
        comm_costs = []
        for op in self.z3_graph.values():
            # no need to include output, since output can be obtained from inputs
            inputs, constraints = op.generate_input_z3()
            for inp in inputs:
                bitvecs[str(inp)] = inp
            # input constraints
            input_constraints.extend(constraints)
            # communication cost
            comm_costs.append(op.calculate_comm_cost_z3())

        reshard_costs = []
        for op in self.z3_graph.values():
            assert (
                len(op.args) <= 1
            ), f"only support single input, but got multiple ({len(op.args)})"
            if not isinstance(op, MatmulOp):
                continue
            for arg in op.args:
                curr = bitvecs[op.name + "_lhs"]
                prev = arg.generate_output_z3()
                reshard_costs.append(
                    self.calculate_reshard_cost_z3(prev, curr, arg.out_size)
                )
            # final output should not be sharded
            if len(op.users) == 0:
                next_inp = ShardSpec("RR").id
                reshard_costs.append(
                    self.calculate_reshard_cost_z3(
                        op.generate_output_z3(), next_inp, op.out_size
                    )
                )

        self.cost = sum(comm_costs) + sum(reshard_costs)
        self.goal += input_constraints

    def solve(self, inputs, max_iter=100):
        self.inference_shape(inputs)
        self.construct_z3_graph()
        self.construct_z3_problem()
        sol = z3.Solver()
        sol.add(self.goal)
        max_cost = 1e12
        for it in range(max_iter):
            print(f"=================== Iter {it} ===================")
            sol.push()
            assert self.cost is not None
            sol.add(self.cost < max_cost)
            # print(sol)
            sat = sol.check()
            if str(sat) == "unsat":
                print("Cannot find better solutions")
                break
            mod = sol.model()
            print(mod)
            results = {d.name(): mod[d] for d in mod.decls()}
            max_cost = 0
            for name, op in self.z3_graph.items():
                if not isinstance(op, MatmulOp):
                    continue
                lhs = results[f"{name}_lhs"]
                rhs = results[f"{name}_rhs"]
                op.set_concrete_values(lhs, rhs)
                output = op.generate_output()
                print(f"{name}: {op.lhs_shape} x {op.rhs_shape} = {op.out_shape}")
                print(
                    f"  {name}: {ShardSpec(lhs)} x {ShardSpec(rhs)} = {ShardSpec(output)}"
                )
                comm_cost = op.calculate_comm_cost()
                max_cost += comm_cost
                print(f"  Comm cost: {comm_cost}")
                for arg in op.args:
                    curr = lhs
                    prev = arg.generate_output()
                    reshard_cost = self.calculate_reshard_cost(prev, curr, arg.out_size)
                    max_cost += reshard_cost
                    print(
                        f"  Resharding cost ({arg.name}) {ShardSpec(prev)} -> {ShardSpec(curr)}: {reshard_cost}"
                    )
                if len(op.users) == 0:
                    next_inp = ShardSpec("RR").id
                    reshard_cost = self.calculate_reshard_cost(
                        output, next_inp, op.out_size
                    )
                    max_cost += reshard_cost
                    print(
                        f"  Last resharding cost {ShardSpec(output)} -> {ShardSpec(next_inp)}: {reshard_cost}"
                    )
            print("Total cost:", max_cost)
            sol.pop()
        # generate sharding sequence
        self.best_spec = results
        print()
        print("Best solution:")
        for name, op in self.z3_graph.items():
            if not isinstance(op, MatmulOp):
                continue
            weight = self.best_spec[f"{name}_rhs"]
            if weight == ShardSpec("RS").id:
                dim = 0  # transposed
            elif weight == ShardSpec("SR").id:
                dim = 1
            else:
                continue
            if op.node.op == "call_module":
                print(f'sch["{op.node.target}"].shard("weight", dim={dim})')
                if dim == 0:
                    print(f'sch["{op.node.target}"].shard("bias", dim={dim})')
                if (
                    self.best_spec[f"{name}_lhs"] == ShardSpec("RS").id
                    and self.best_spec[f"{name}_rhs"] == ShardSpec("SR").id
                ):
                    print(
                        f'sch["{op.node.target}"].sync(mode="fwd_post", sync_op_or_fn="all_reduce")'
                    )
