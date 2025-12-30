import random


# 工具函数：扩展欧几里得算法求逆元
def mod_inverse(a, m):
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0
    while a > 1:
        # q is quotient
        q = a // m
        t = m
        # m is remainder now, process same as Euclid's algo
        m = a % m
        a = t
        t = y
        # Update x and y
        y = x - q * y
        x = t
    if x < 0:
        x = x + m0
    return x


# 工具函数：快速幂
def power(base, exp, mod):
    return pow(base, exp, mod)


class ElGamalZKP:
    def __init__(self, bit_length=128):
        # 实际应用中应使用更大的素数，这里为了演示方便或根据库生成
        # 这里硬编码一个小的安全素数用于演示，或者使用简单逻辑生成
        # p = 2*q + 1
        self.p = 23981  # 示例素数
        self.g = 2  # 生成元
        self.x = random.randint(1, self.p - 2)  # 私钥
        self.h = power(self.g, self.x, self.p)  # 公钥

    def get_pub_key(self):
        return (self.p, self.g, self.h)

    def encrypt(self, m):
        # 随机数 r
        r = random.randint(1, self.p - 2)
        self.last_r = r  # 保存r用于生成ZKP

        c1 = power(self.g, r, self.p)
        c2 = (m * power(self.h, r, self.p)) % self.p
        return (c1, c2)

    def decrypt(self, c1, c2):
        s = power(c1, self.x, self.p)
        m = (c2 * mod_inverse(s, self.p)) % self.p
        return m

    # --- ZKP Prover 角色 ---

    def zkp_commit(self):
        """第一步：承诺"""
        self.k = random.randint(1, self.p - 2)  # 临时随机数
        u1 = power(self.g, self.k, self.p)
        u2 = power(self.h, self.k, self.p)
        return (u1, u2)

    def zkp_response(self, challenge_e):
        """第三步：响应"""
        # s = k + e * r (mod p-1)
        # 注意：指数运算是在模 p-1 群下的
        s = (self.k + challenge_e * self.last_r) % (self.p - 1)
        return s

    # --- ZKP Verifier 角色 ---

    @staticmethod
    def verify(pub_key, cipher, claimed_m, commitment, challenge_e, response_s):
        """第四步：验证"""
        p, g, h = pub_key
        c1, c2 = cipher
        u1, u2 = commitment

        # 验证等式 1: g^s == u1 * c1^e
        left1 = power(g, response_s, p)
        right1 = (u1 * power(c1, challenge_e, p)) % p

        # 验证等式 2: h^s == u2 * (c2/m)^e
        # 计算 c2/m -> c2 * m^-1
        m_inv = mod_inverse(claimed_m, p)
        val = (c2 * m_inv) % p

        left2 = power(h, response_s, p)
        right2 = (u2 * power(val, challenge_e, p)) % p

        print(f"[*] Verifying Equation 1: {left1} == {right1}")
        print(f"[*] Verifying Equation 2: {left2} == {right2}")

        return left1 == right1 and left2 == right2


# --- 主程序测试 ---
if __name__ == "__main__":
    print("=== ElGamal ZKP 实验 ===")

    # 1. 初始化
    eg = ElGamalZKP()
    pub = eg.get_pub_key()
    print(f"公钥 (p, g, h): {pub}")

    # 2. 加密
    msg = 123
    print(f"原始信息: {msg}")
    ciphertext = eg.encrypt(msg)
    print(f"密文 (c1, c2): {ciphertext}")

    # 3. 验证解密（可选）
    decrypted = eg.decrypt(*ciphertext)
    print(f"解密消息: {decrypted}")

    # --- 开始零知识证明 ---
    print("\n--- 零知识证明协议 ---")

    # Step 1: Prover Commit
    u_vals = eg.zkp_commit()
    print(f"1. Prover发出承诺 (u1, u2): {u_vals}")

    # Step 2: Verifier Challenge
    e = random.randint(1, pub[0] - 2)
    print(f"2. Verifier发出挑战 e: {e}")

    # Step 3: Prover Response
    s = eg.zkp_response(e)
    print(f"3. Prover发出响应 s: {s}")

    # Step 4: Verification
    # 假设Prover声称密文对应的明文确实是 msg
    is_valid = ElGamalZKP.verify(pub, ciphertext, msg, u_vals, e, s)

    if is_valid:
        print("\n[RESULT] 验证成功! Prover 知道 r 和 m 是正确的")
    else:
        print("\n[RESULT] 验证失败!")

    # --- 模拟攻击测试 (错误的明文) ---
    print("\n--- 模拟攻击测试 (错误的明文) ---")
    fake_msg = 456
    # 证明者试图证明密文对应的是 456 (实际上是 123)
    # 即使 s 是正确计算的，验证也会失败，因为 c2/fake_msg 不等于 h^r
    is_valid_fake = ElGamalZKP.verify(pub, ciphertext, fake_msg, u_vals, e, s)
    if not is_valid_fake:
        print("[RESULT] 假消息验证如预期失败")
