from tabulate import tabulate


def custom_log(msg):
    print(msg)


def custom_log_tuple(a, b):
    print(tabulate([[a, b]], tablefmt="simple_grid"))
