from skimage import color
import colour
import timeit
import deltae


p1 = [52.96599120872003, 18.293357903311215, -65.41748712308582]
p2 = [0.19466490621016064, 0.9236630588433103, -0.10216916192304915]


def delta_sk():
    return color.deltaE_ciede2000(p1, p2)


def delta_science():
    return colour.delta_E(p1, p2, method="CIE 2000")


def delta_deltae():
    p1_dict = {"L": p1[0], "a": p1[1], "b": p1[2]}
    p2_dict = {"L": p2[0], "a": p2[1], "b": p2[2]}
    return deltae.delta_e_2000(p1_dict, p2_dict)


print("test")
print("delta_sk:", delta_sk())
print("delta_science", delta_science())
print("delta_deltae", delta_deltae())
print(timeit.timeit(delta_sk, number=100000))
print(timeit.timeit(delta_science, number=100000))
print(timeit.timeit(delta_deltae, number=100000))
