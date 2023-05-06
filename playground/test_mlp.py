import z3

"""
R: Replicated
S: Sharded
0: 00 RR
1: 01 RS
2: 10 SR
"""

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

goal = []
goal += input_constraints
goal += format_constraints

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
for op in ops:
    lhs = results[f"{op}_lhs"]
    rhs = results[f"{op}_rhs"]
    print(f"{op}: {Spec(lhs)} x {Spec(rhs)}")
