import z3

"""
R: Replicated
S: Sharded
0: 00 RR
1: 01 RS
2: 10 SR
"""

# kernel size
M = 512
K = 1024
N = 1024
p = 2

bitvecs = {}

ops = ["matmul", "matmul_1"]

class Spec:
    def __init__(self, spec):
        self.map = {
            "RR": 0,
            "RS": 1,
            "SR": 2
        }
        if isinstance(spec, str):
            self.spec = spec
        else:
            self.spec = list(self.map.keys())[list(self.map.values()).index(spec)]

    @property
    def id(self):
        return self.map[self.spec]

    def __str__(self):
        return self.spec

output_map = {"RR": "RS", "RS": "RR", "SR": "SR"}

class Matmul:

    @staticmethod
    def generate_output(lhs, rhs):
        return Spec(output_map[Spec(lhs).spec]).id
    
    @staticmethod
    def generate_output_z3(lhs, rhs):
        result = 3 # invalid
        for inp, out in output_map.items():
            result = z3.If(lhs == Spec(inp).id, Spec(out).id, result)
        return result


input_constraints = []
format_constraints = []
outputs = []
comm_costs = []
for op in ops:
    # no need to include output, since output can be obtained from inputs
    lhs = z3.BitVec(f"{op}_lhs", 2) # input
    rhs = z3.BitVec(f"{op}_rhs", 2) # weight
    bitvecs[f"{op}_lhs"] = lhs
    bitvecs[f"{op}_rhs"] = rhs

    # input constraints
    constraints = [z3.And(lhs == Spec("RR").id, rhs == Spec("RS").id),
                  z3.And(lhs == Spec("RS").id, rhs == Spec("SR").id),
                  z3.And(lhs == Spec("SR").id, rhs == Spec("RR").id)]
    input_constraints.append(z3.Or(*constraints))

    # format constraints
    format_constraints.extend([z3.ULE(lhs, 3), z3.ULE(rhs, 3)])

    # output
    outputs.append(Matmul.generate_output_z3(lhs, rhs)) # SR x RR

    # communication cost
    comm_costs.append(z3.If(lhs == Spec("RR").id, # RR x RS
                     0,
                 z3.If(lhs == Spec("RS").id, # RS x SR
                      M * K if op == "matmul" else K * N,
                      0))) # SR x RR

def calculate_reshard_cost(prev, curr):
    return z3.If(prev == Spec("RR").id,
        z3.If(curr == Spec("RR").id, 0,
            z3.If(curr == Spec("RS").id, 0,
                z3.If(curr == Spec("SR").id, 0, 0))),
        z3.If(prev == Spec("RS").id,
            z3.If(curr == Spec("RR").id, 1 / p * M * K,
                z3.If(curr == Spec("RS").id, 0,
                    z3.If(curr == Spec("SR").id, (1 / p - 1 / (p * p)) * M * K, 0))),
            z3.If(prev == Spec("SR").id,
                z3.If(curr == Spec("RR").id, 1 / p * M * K,
                    z3.If(curr == Spec("RS").id, (1 / p - 1 / (p * p)) * M * K,
                        z3.If(curr == Spec("SR").id, 0, 0))),
                0)))

reshard_costs = []
for i in range(len(outputs)):
    prev = outputs[i]
    curr = outputs[i + 1] if i < len(outputs) - 1 else Spec("RR").id
    reshard_costs.append(calculate_reshard_cost(prev, curr))

cost = sum(comm_costs) + sum(reshard_costs)

goal = []
goal += [bitvecs["matmul_lhs"] == Spec("RR").id]
goal += input_constraints
goal += format_constraints

def model_values(model):
    return {
        d.name(): model[d]
        for d in model.decls()
    }

def calculate_comm_cost(lhs, rhs):
    if lhs == Spec("RR").id: # RR x RS
        return 0
    elif lhs == Spec("RS").id: # RS x SR
        return M * K # all_reduce
    elif lhs == Spec("SR").id:
        return 0

def calculate_reshard_cost(prev, curr):
    print("reshard", prev, curr)
    if prev == Spec("RR").id:
        if curr == Spec("RR").id:
            return 0
        elif curr == Spec("RS").id:
            return 0
        elif curr == Spec("SR").id:
            return 0
    elif prev == Spec("RS").id:
        if curr == Spec("RR").id:
            return 1 / p * M * K
        elif curr == Spec("RS").id:
            return 0
        elif curr == Spec("SR").id:
            return (1 / p - 1 / (p * p)) * M * K
    elif prev == Spec("SR").id:
        if curr == Spec("RR").id:
            return 1 / p * M * K
        elif curr == Spec("RS").id:
            return (1 / p - 1 / (p * p)) * M * K
        elif curr == Spec("SR").id:
            return 0
    else:
        raise ValueError("Invalid spec")

sol = z3.Solver()
max_cost = 10e8
for _ in range(3):
    print("=====================================")
    sol.add(goal)
    sol.push()
    sol.add(cost < max_cost)
    sol.check()
    mod = sol.model()
    print(mod)
    results = model_values(mod)
    print(calculate_reshard_cost(Spec("RR").id, results[f"{ops[0]}_lhs"]))
    max_cost = 0
    for i, op in enumerate(ops):
        lhs = results[f"{op}_lhs"]
        rhs = results[f"{op}_rhs"]
        output = Matmul.generate_output(lhs, rhs)
        print(f"{op}: {Spec(lhs)} x {Spec(rhs)} = {Spec(output)}")
        comm_cost = calculate_comm_cost(lhs, rhs)
        reshard_cost = calculate_reshard_cost(output, results[f"{ops[i + 1]}_lhs"] if i < len(ops) - 1 else Spec("RR").id)
        max_cost += comm_cost + reshard_cost
        print(comm_cost, reshard_cost)
    print("Total cost:", max_cost)
    sol.pop()
