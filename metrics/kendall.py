import itertools as it

def kendallTau(A, B=None):
    # if any partial is B
    if B is None : B = list(range(len(A)))
    n = len(A)
    pairs = it.combinations(range(n), 2)
    distance = 0
    # print("IIIIMNNMNNN",list(pairs),len(A))
    for x, y in pairs:
        #if not A[x]!=A[x] and not A[y]!=A[y]:#OJO no se check B
        a = A[x] - A[y]
        try:
            b = B[x] - B[y]# if discordant (different signs)
        except:
            print("ERROR kendallTau, check b",A, B, x, y)
        # print(b,a,b,A, B, x, y,a * b < 0)
        if (a * b < 0):
            distance += 1
    return distance


def kendalltau_partial_both(a,b,k):
    a_top = a.copy()
    b_top = b.copy()
    if k>0:
        a_top[a_top > k] = k + 1
        b_top[b_top > k] = k + 1
        #maxdist = k * (k - 1) / 2 + k * (len(a) - k)
        maxdist = k*k
    else:
        maxdist = len(a)*(len(a)-1)/2

    dist = kendallTau(a_top,b_top)
    return 1- dist/maxdist