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

input_constraints = []
format_constraints = []
outputs = []
comm_costs = []
for op in ops:
    # no need to include output, since output can be obtained from inputs
    lhs = z3.BitVec(f"{op}_lhs", 2) # input
    rhs = z3.BitVec(f"{op}_rhs", 2) # weight
    bitvecs[lhs] = lhs
    bitvecs[rhs] = rhs

    # input constraints
    constraints = [z3.And(lhs == Spec("RR").id, rhs == Spec("RS").id),
                  z3.And(lhs == Spec("RS").id, rhs == Spec("SR").id),
                  z3.And(lhs == Spec("SR").id, rhs == Spec("RR").id)]
    input_constraints.append(z3.Or(*constraints))

    # format constraints
    format_constraints.extend([z3.ULE(lhs, 3), z3.ULE(rhs, 3)])

    # output
    outputs.append(z3.If(lhs == Spec("RR").id, # RR x RS
                   Spec("RS").id,
                z3.If(lhs == Spec("RS").id, # RS x SR
                    Spec("RR").id,
                    Spec("SR").id))) # SR x RR

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
goal += input_constraints
goal += format_constraints
goal.append(cost < 1e10)

def solve(phi):
    sol = z3.Solver()
    sol.add(phi)
    sol.check()
    return sol.model()

def model_values(model):
    return {
        d.name(): model[d]
        for d in model.decls()
    }

mod = solve(goal)
print(mod)
results = model_values(mod)
for i, op in enumerate(ops):
    lhs = results[f"{op}_lhs"]
    rhs = results[f"{op}_rhs"]
    # output = generate_output(lhs, rhs)
    print(f"{op}: {Spec(lhs)} x {Spec(rhs)}")
    # print(calculate_comm_cost(lhs, rhs), calculate_reshard_cost(output, results[f"{ops[i + 1]}_lhs"] if i < len(ops) - 1 else Spec("RR").id))
