# xor 회로는 NAND + OR + AND 회로를 이용하여 구현할 수 있음

def XOR(x1,x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    s3 = AND(s1,s2)
    return s3

def AND(x1, x2):
    if x1==1 and x2==1:
        return 1
    else: return 0

def OR(x1, x2):
    if x1==0 and x2==0:
        return 0
    else: return 1

def NAND(x1, x2):
    if x1==1 and x2==1:
        return 0
    else: return 1

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))