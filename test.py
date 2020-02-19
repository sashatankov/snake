


class A:
    x = 6

    def __init__(self):
        self.y = 4

    def foo(self):
        self.z = 9
        A.x = 7
        print(A.x)


def main():
    a = A()
    a.foo()
    print(a.z)


if __name__ == '__main__':
    main()