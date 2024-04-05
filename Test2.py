from BFV import *
from helper import *
from random import randint
from math import log,ceil

# Parameter generation (pre-defined or generate parameters)
PD = 0 # 0: generate -- 1: pre-defined

if PD == 0:
    t = 256;  n, q, psi = 2048 , 137438691329        , 22157790             # log(q) = 37

    # other necessary parameters
    psiv= modinv(psi,q)
    w   = pow(psi,2,q)
    wv  = modinv(w,q)
else:
    t, n, logq = 256, 2048, 37
    q,psi,psiv,w,wv = ParamGen(n,logq)

# Determine mu, sigma (for discrete gaussian distribution)
mu    = 0
sigma = 0.5 * 3.2

# Determine T, p (for relinearization and galois keys) based on noise analysis
T = 256
p = q**3 + 1

# Generate polynomial arithmetic tables
w_table    = [1]*n
wv_table   = [1]*n
psi_table  = [1]*n
psiv_table = [1]*n
for i in range(1,n):
    w_table[i]    = ((w_table[i-1]   *w)    % q)
    wv_table[i]   = ((wv_table[i-1]  *wv)   % q)
    psi_table[i]  = ((psi_table[i-1] *psi)  % q)
    psiv_table[i] = ((psiv_table[i-1]*psiv) % q)

qnp = [w_table,wv_table,psi_table,psiv_table]

print("--- Starting BFV Demo")

# Generate BFV evaluator
Evaluator1 = BFV(n, q, t, mu, sigma, qnp)
Evaluator2 = BFV(n, q, t, mu, sigma, qnp)

# Generate Keys
Evaluator1.SecretKeyGen()
Evaluator2.SecretKeyGen()
a = Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp)
a.randomize(Evaluator1.q)
# PK for E1
e1 = Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp)
e1.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator1.sigma)
pk0 = -(a * Evaluator1.sk + e1)
pk1 = a
Evaluator1.pk = [pk0, pk1]
# PK for E2
e2 = Poly(Evaluator2.n, Evaluator2.q, Evaluator2.qnp)
e2.randomize(0, domain=False, type=1, mu=Evaluator2.mu, sigma=Evaluator2.sigma)
pk0 = -(a * Evaluator2.sk + e2)
pk1 = a
Evaluator2.pk = [pk0, pk1]

# Aggregate PK
"""
Aggregated PK = sum(PKi) = sum(-si*a) + sum(ei) mod q
"""
aggpk = -Evaluator1.sk*a + -Evaluator2.sk*a + (e1+e2)

# Generate message
n1, n2 = 15, -5
print("---integers n1 and n2 are generated.")
print("* n1: {}".format(n1))
print("* n2: {}".format(n2))
print("* n1+n2: {}".format(n1+n2))
print("")

# Encode random messages into plaintext polynomials
print("--- n1 and n2 are encoded as polynomials m1(x) and m2(x).")
m1 = Evaluator1.IntEncode(n1)
m2 = Evaluator2.IntEncode(n2)
print("* m1(x): {}".format(m1))
print("* m2(x): {}".format(m2))
print("")

# Encrypt message
ct1 = Evaluator1.Encryption1(m1,aggpk)
# u1 = Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp)
#u1.randomize(2)
#e10, e11 = Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp) , Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp)
#e10.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator1.sigma)
#e11.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator1.sigma)
#c0 = (aggpk * u1 + e10 + m1)
#c1 = (a * u1 + e11) % q
#ct1 = [c0,c1]

ct2 = Evaluator2.Encryption1(m2,aggpk)
#u2 = Poly(Evaluator2.n, Evaluator2.q, Evaluator2.qnp)
#u2.randomize(2)
#e20,e21 = Poly(Evaluator2.n, Evaluator2.q, Evaluator2.qnp),Poly(Evaluator2.n, Evaluator2.q, Evaluator2.qnp)
#e20.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator2.sigma)
#e21.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator1.sigma)
#c0 = (aggpk * u2 + e20 + m2) % q
#c1 = (a * u2 + e21) % q
#ct2 = [c0,c1]

print("--- m1 and m2 are encrypted as ct1 and ct2.")
print("* ct1[0]: {}".format(ct1[0]))
print("* ct1[1]: {}".format(ct1[1]))
print("* ct2[0]: {}".format(ct2[0]))
print("* ct2[1]: {}".format(ct2[1]))
print("")

# Homomorphic Addition
ct0_b = ct1[0] + ct2[0]
ct1_b = ct1[1] + ct2[1]
ct = [ct0_b, ct1_b]

# Decrypt message
e = Poly(Evaluator1.n, Evaluator1.q, Evaluator1.qnp)
e.randomize(0, domain=False, type=1, mu=Evaluator1.mu, sigma=Evaluator1.sigma)
D1 = Evaluator1.sk * ct[1] + e
D2 = Evaluator2.sk * ct[1] + e
mt = ct[0] + D1+D2

nr = Evaluator1.IntDecode(mt)
ne = (n1+n2)

print("--- Performing ct_add = Enc(m1) + Enc(m2)")
print("* ct_add[0] :{}".format(ct[0]))
print("* ct_add[1] :{}".format(ct[1]))
print("--- Performing ct_dec = Dec(ct_add)")
print("* ct_dec    :{}".format(mt))
print("--- Performing ct_dcd = Decode(ct_dec)")
print("* ct_dcd    :{}".format(nr))

if nr == ne:
    print("* Homomorphic addition works.")
else:
    print("* Homomorphic addition does not work.")
print("")