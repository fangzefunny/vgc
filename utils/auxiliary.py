import itertools

def get_all_construals(chars='0123456'):
    
    # define the characters that can be used
    chars = list(chars)

    # generate all possible permutations of length 1 to 7
    perms = []
    for length in range(1, 8):
        perms += itertools.permutations(chars, length)

    # convert the permutations to strings
    perms_as_str = [''.join(perm) for perm in perms]

    # print the resulting strings
    lst = []
    for perm in perms_as_str:
        p = ''.join(sorted(perm))
        if p not in lst: lst.append(p)
    lst += ['']

    return lst 