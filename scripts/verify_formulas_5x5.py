"""Owner order: the two to-be-added formulas checked FIVE
ways against math + authorities, then FIVE real-number
calculations each (all-different numbers).
 Formula A (entropy grad): p=softmax(z), H=-sum p ln p,
   dH/dz_k = -p_k (ln p_k + H)
 Formula B (KL grad): q=softmax(z_ref) fixed,
   KL = sum p (ln p - ln q),
   dKL/dz_k = p_k [ (ln p_k - ln q_k) - KL ]
"""
import numpy as np, torch, sympy as sp
from mpmath import mp, mpf, exp as mexp, log as mlog
mp.dps = 50
ok_all = True
def rep(name, ok, detail=""):
    global ok_all; ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

def np_softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()
def gradH(z):
    p = np_softmax(z); H = -(p*np.log(p)).sum()
    return -p*(np.log(p)+H)
def gradKL(z, zr):
    p, q = np_softmax(z), np_softmax(zr)
    kl = (p*(np.log(p)-np.log(q))).sum()
    return p*((np.log(p)-np.log(q)) - kl)

# ===== 檢查 1/5:sympy 符號推導(K=3 通式) =====
zs = sp.symbols('z0 z1 z2'); K = 3
es = [sp.exp(z) for z in zs]; Z = sum(es)
ps = [e/Z for e in es]
H = -sum(p*sp.log(p) for p in ps)
ok = True
for k in range(K):
    analytic = -ps[k]*(sp.log(ps[k]) + H)
    ok &= sp.simplify(sp.diff(H, zs[k]) - analytic) == 0
rep("check1 sympy 符號:dH/dz == -p(ln p + H)", bool(ok))
zr = sp.symbols('r0 r1 r2')
qs = [sp.exp(r)/sum(sp.exp(rr) for rr in zr) for r in zr]
KL = sum(p*(sp.log(p)-sp.log(q)) for p, q in zip(ps, qs))
ok = True
for k in range(K):
    analytic = ps[k]*((sp.log(ps[k])-sp.log(qs[k])) - KL)
    ok &= sp.simplify(sp.diff(KL, zs[k]) - analytic) == 0
rep("check1 sympy 符號:dKL/dz == p[(ln p - ln q) - KL]", bool(ok))

# ===== 檢查 2/5:torch autograd 逐位對照 =====
r = np.random.default_rng(101)
okH = okK = True
for _ in range(3):
    z = torch.tensor(r.normal(size=5), requires_grad=True)
    zr_ = torch.tensor(r.normal(size=5))
    p = torch.softmax(z, 0); Ht = -(p*torch.log(p)).sum()
    Ht.backward()
    okH &= np.allclose(z.grad.numpy(), gradH(z.detach().numpy()), atol=1e-12)
    z2 = torch.tensor(z.detach().numpy(), requires_grad=True)
    p2 = torch.softmax(z2, 0); q2 = torch.softmax(zr_, 0)
    KLt = (p2*(torch.log(p2)-torch.log(q2))).sum()
    KLt.backward()
    okK &= np.allclose(z2.grad.numpy(), gradKL(z2.detach().numpy(), zr_.numpy()), atol=1e-12)
rep("check2 torch autograd:熵梯度逐位一致(3 例)", okH)
rep("check2 torch autograd:KL 梯度逐位一致(3 例)", okK)

# ===== 檢查 3/5:mpmath-50 中心差分 =====
def fdH(z, h=mpf('1e-20')):
    z = [mpf(str(x)) for x in z]
    def Hf(zz):
        es = [mexp(x) for x in zz]; Z = sum(es)
        return -sum((e/Z)*mlog(e/Z) for e in es)
    g = []
    for k in range(len(z)):
        zp = list(z); zm = list(z)
        zp[k]+=h; zm[k]-=h
        g.append(float((Hf(zp)-Hf(zm))/(2*h)))
    return np.array(g)
z = np.array([0.3, -1.1, 0.7, 2.2])
rep("check3 mpmath-50 FD:熵梯度", np.allclose(fdH(z), gradH(z), atol=1e-12),
    f"max diff {np.max(np.abs(fdH(z)-gradH(z))):.1e}")
def fdKL(z, zr, h=mpf('1e-20')):
    z = [mpf(str(x)) for x in z]; zr = [mpf(str(x)) for x in zr]
    def KLf(zz):
        es = [mexp(x) for x in zz]; Z = sum(es)
        er = [mexp(x) for x in zr]; Zr = sum(er)
        return sum((e/Z)*(mlog(e/Z)-mlog(f/Zr)) for e, f in zip(es, er))
    g = []
    for k in range(len(z)):
        zp = list(z); zm = list(z)
        zp[k]+=h; zm[k]-=h
        g.append(float((KLf(zp)-KLf(zm))/(2*h)))
    return np.array(g)
zr2 = np.array([1.0, 0.2, -0.5, 0.9])
rep("check3 mpmath-50 FD:KL 梯度", np.allclose(fdKL(z, zr2), gradKL(z, zr2), atol=1e-12),
    f"max diff {np.max(np.abs(fdKL(z, zr2)-gradKL(z, zr2))):.1e}")

# ===== 檢查 4/5:SB3 源碼語義(熵項在 loss 裡的符號/位置) =====
import inspect, stable_baselines3.ppo.ppo as sbppo
src = inspect.getsource(sbppo.PPO.train)
okS = ("entropy_loss = -th.mean(entropy)" in src and
       "+ self.ent_coef * entropy_loss" in src)
rep("check4 SB3 源碼:loss = policy + ent_coef*(-H) + vf*value(符號一致)", okS)
# 對應到我們的 gz 約定:d/dz [ -ent_coef*H ] = +ent_coef * (-dH/dz)*(-1)?
# 直接數值定案:SB3 等效 torch loss 對 logits 的梯度 == 我們計劃的
# gz_ent = ent_coef * ( -gradH ) ... 用 torch 全式驗證:
z = torch.tensor(r.normal(size=4), requires_grad=True)
ec = 0.37
p = torch.softmax(z, 0); H = -(p*torch.log(p)).sum()
loss = ec * (-H)                     # SB3 的 ent_coef*entropy_loss
loss.backward()
plan_gz = ec * (-gradH(z.detach().numpy()))
rep("check4 SB3 語義數值定案:計劃 gz_ent == autograd(ent_coef*(-H))",
    np.allclose(z.grad.numpy(), plan_gz, atol=1e-12))

# ===== 檢查 5/5:獨立 numpy 實現(第二人重寫式)交叉對照 =====
def gradH_alt(z):
    p = np_softmax(z)
    J = np.diag(p) - np.outer(p, p)          # softmax Jacobian
    dH_dp = -(np.log(p) + 1.0)
    return J @ dH_dp
def gradKL_alt(z, zr):
    p, q = np_softmax(z), np_softmax(zr)
    J = np.diag(p) - np.outer(p, p)
    dKL_dp = np.log(p) - np.log(q) + 1.0
    return J @ dKL_dp
z5 = r.normal(size=6); zr5 = r.normal(size=6)
rep("check5 獨立實現(雅可比路線):熵梯度一致",
    np.allclose(gradH(z5), gradH_alt(z5), atol=1e-12))
rep("check5 獨立實現(雅可比路線):KL 梯度一致",
    np.allclose(gradKL(z5, zr5), gradKL_alt(z5, zr5), atol=1e-12))

# ===== 驗算 5 遍:真實數字,每遍不同(維度/數值全不同) =====
print("\n===== 真實數字驗算 5 遍(熵梯度 + KL 梯度,逐遍打印) =====")
cases = [
    np.array([0.5, -0.2]),                       # K=2
    np.array([1.0, 2.0, 3.0]),                   # K=3 整數
    np.array([-0.7, 0.0, 0.4, 1.3]),             # K=4 含零
    np.array([2.5, -3.1, 0.05, 1.8, -0.6]),      # K=5 大範圍
    np.array([0.111, 0.222, 0.333, 0.444, 0.555, 0.666]),  # K=6 等差
]
refs = [
    np.array([-0.3, 0.9]),
    np.array([3.0, 2.0, 1.0]),
    np.array([0.2, 0.2, 0.2, 0.2]),
    np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
    np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
]
for i, (z, zr_) in enumerate(zip(cases, refs), 1):
    gH, gH_fd = gradH(z), fdH(z)
    gK, gK_fd = gradKL(z, zr_), fdKL(z, zr_)
    t = torch.tensor(z, requires_grad=True)
    p = torch.softmax(t, 0); (-(p*torch.log(p)).sum()).backward()
    oki = (np.allclose(gH, gH_fd, atol=1e-12)
           and np.allclose(gH, t.grad.numpy()*-1*-1, atol=1e-12)
           and np.allclose(gK, gK_fd, atol=1e-12))
    print(f" 驗算{i}: z={np.round(z,3).tolist()}")
    print(f"   dH/dz  = {np.round(gH,10).tolist()}")
    print(f"   FD 差  = {np.max(np.abs(gH-gH_fd)):.1e} | torch 差 = "
          f"{np.max(np.abs(gH - t.grad.numpy())):.1e}")
    print(f"   dKL/dz = {np.round(gK,10).tolist()}  FD 差 = "
          f"{np.max(np.abs(gK-gK_fd)):.1e}")
    rep(f"驗算{i}", bool(oki))
print("\nTOTAL:", "ALL PASS" if ok_all else "FAILURES PRESENT")
