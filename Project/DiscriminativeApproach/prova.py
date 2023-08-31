import itertools

if __name__=="__main__":
    keys1 = ['10^-5', '10^-4', '10^-3', '10^-2', '10^-1', '1', '10^1', '10^2', '10^3', '10^4', '10^5']
    keys2 = ['10^-3', '10^-2', '10^-1']

    keys = list(itertools.product(keys1, keys2))
    print(keys)