from fractions import Fraction
import numpy as np
def solution(m):
    size = len(m)
    # Identify terminal and non-terminal states
    terminal_states = [i for i, row in enumerate(m) if sum(row) == 0]
    non_terminal_states = [i for i in range(size) if i not in terminal_states]
    # Rearrange the matrix
    rearranged_matrix = [row[:size] + row[size:] for row in m]
    print("rearranged_matrix:", rearranged_matrix)
    # Normalize the rows of the matrix
    normalized_matrix = [[Fraction(x, sum(row)) if sum(row) != 0 else Fraction(0) for x in row] for row in rearranged_matrix]
    # Split the matrix into submatrices Q, R
    Q = [row[:len(non_terminal_states)] for row in normalized_matrix[:len(non_terminal_states)]]
    R = [row[len(non_terminal_states):] for row in normalized_matrix[:len(non_terminal_states)]]
    # Calculate the fundamental matrix F
    Q_np = np.array(Q, dtype=float) # Convert Q to a numpy array
    print("Q_np",Q_np)
    I_minus_Q_inverse = np.linalg.inv(np.subtract(np.identity(len(Q_np)), Q_np))
    print("I_minus_Q_inverse",I_minus_Q_inverse)
    # Convert to fractions
    F = [[Fraction(float(num)).limit_denominator() for num in row] for row in I_minus_Q_inverse.tolist()] #--<
    print("F:",F)

    fr_subm = np.dot(I_minus_Q_inverse, R)
    print("Fr:",fr_subm)
    # Calculate the absorption matrix B
    B = [[sum(F[i][k] * R[k][j] for k in range(len(F))).limit_denominator() for j in range(len(R[0]))] for i in range(len(F))] #--<
    print("B",B)
    # Find the common denominator
    denominator = max(fraction.denominator for fraction in B[0])
    print("Den",denominator)
    # Convert the result to the required format
    result = [fraction.numerator * (denominator // fraction.denominator) for fraction in B[0] ] + [denominator]
    return result

# Example usage:
m = [[0, 1, 0, 0, 0, 1], [4, 0, 0, 3, 2, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]

result = solution(m)
print(result)


def solution(m):
    # case when 0 is terminal state
    if (not any(m[0])):
        return [1] + ([0] * (len(m) - 1)) + [1]

    # diagonal values arent relevant
    for i in range(len(m)):
        m[i][i] = 0

    probabilities_matrix = []
    for i in range(len(m)):
        probabilities_matrix.append([None] * len(m))
        for j in range(len(m)):
            probabilities_matrix[i][j] = frac_div([m[i][j], 1], [sum(m[i]), 1])
    terminals = []
    not_terminals = list(range(1, len(m)))
    for i in range(len(m)):
        if (not any(m[i])):
            terminals.append(i)
            not_terminals.remove(i)

    # remove not-terminals nodes
    for i in not_terminals:
        absorb_node(probabilities_matrix, i)


    terminals_probabilities = list(map(lambda x: probabilities_matrix[0][x], terminals))
    common_denominator = get_common_denominator(list(map(lambda x: x[1], terminals_probabilities)))
    unsimplified_numerators = list(map(lambda x: frac_unsimplify(x, common_denominator)[0], terminals_probabilities))

    return unsimplified_numerators + [common_denominator]


def absorb_node(pm, node):
    for i in range(len(pm)):
        for j in range(len(pm)):
            if (i != node and j != node):
                pm[i][j] = frac_add(pm[i][j], frac_mult(pm[i][node], pm[node][j]))

    for k in range(len(pm)):
        pm[k][node] = [0, 1]
        pm[node][k] = [0, 1]

    for i in range(len(pm)):
        if (pm[i][i] != [0, 1]):
            multiplier = solve_geometric_series(pm[i][i])
            for j in range(len(pm)):
                if (i == j):
                    pm[i][j] = [0, 1]
                else:
                    pm[i][j] = frac_mult(pm[i][j], multiplier)


# we will work with fractions, so let's create some functions

def frac_simplify(a):
    if (a[0] == 0):
        a[1] = 1
    i = 2
    while (i <= max(a)):
        if (a[0] % i == 0 and a[1] % i == 0):
            a[0] //= i
            a[1] //= i
        else:
            i += 1
    return a


def frac_add(a, b):
    return frac_simplify([a[0] * b[1] + b[0] * a[1], a[1] * b[1]])


def frac_subs(a, b):
    return frac_simplify([a[0] * b[1] - b[0] * a[1], a[1] * b[1]])


def frac_mult(a, b):
    return frac_simplify([a[0] * b[0], a[1] * b[1]])


def frac_div(a, b):
    if (a[1] == 0 or b[1] == 0):
        return [0, 1]
    return frac_simplify([a[0] * b[1], a[1] * b[0]])


def solve_geometric_series(r):
    if (r == [1, 1]):
        return [1, 1]
    n = [1, 1]
    d = frac_subs([1, 1], r)
    return frac_div(n, d)


def get_common_denominator(l):
    greater = min(l)
    while (not all(list(map(lambda x: greater % x == 0, l)))):
        greater += 1
    return greater


def frac_unsimplify(a, d):
    return [int(a[0] * (d / a[1])), d]