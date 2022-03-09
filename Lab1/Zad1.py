
def prime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def select_primes(x):
    result = x.copy()
    for i in x:
        if not prime(i):
            result.remove(i)
    return result


if __name__ == '__main__':
    print(prime(2))
    print(prime(4))
    print(prime(85))
    print(prime(97))
    print(select_primes([1, 2, 3, 4, 5, 6]))
    print(select_primes([11, 22, 13, 49, 85]))


