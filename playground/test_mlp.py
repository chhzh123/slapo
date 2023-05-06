import z3

"""
R: Replicated
S: Sharded
0: 00 RR
1: 01 RS
2: 10 SR
"""

l1_input = z3.BitVec("L1_input", 2)
l1_weight = z3.BitVec("L1_weight", 2)
# l1_output = z3.BitVec("L1_output", 2)
l2_input = z3.BitVec("L2_input", 2)
l2_weight = z3.BitVec("L2_weight", 2)
# l2_output = z3.BitVec("L2_output", 2)

# input constraints
input1_constraints = z3.Or(
    z3.And(l1_input == 0, l1_weight == 1),
        z3.Or(z3.And(l1_input == 1, l1_weight == 2),
            z3.And(l1_input == 2, l1_weight == 0)))

# output constraints
l1_output = z3.If(l1_input == 0, 1,
    z3.If(l1_input == 1, 0, 2))
l2_output = z3.If(l2_input == 0, 1,
    z3.If(l2_input == 1, 0, 2))

input2_constraints = z3.Or(
    z3.And(l2_input == 0, l2_weight == 1),
        z3.Or(z3.And(l2_input == 1, l2_weight == 2),
            z3.And(l2_input == 2, l2_weight == 0)))

# format constraints
format_constraints = z3.And(z3.ULE(l1_input, 3), z3.ULE(l1_weight, 3), z3.ULE(l2_input, 3), z3.ULE(l2_weight, 3))

goal = [input1_constraints, input2_constraints, format_constraints, l2_input != 2, l1_input == 0]

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
for key, value in results.items():
    if value == 0:
        value = "RR"
    elif value == 1:
        value = "RS"
    elif value == 2:
        value = "SR"
    print(f"{key} = {value}")
