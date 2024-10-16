# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class ITSNetwork:
    # Need Set
    T: int = None
    theta1: float = None  # θ_1 represents the probability that NIU is affected by EIU
    theta2: float = None  # θ_2 represents the probability that NMSP is affected by EIU
    theta3: float = None  # θ_3 represents the probability that NAMSP is affected by EIU
    theta4: float = None  # θ_4 represents the probability that NAMS is affected by NMSP/EMSP
    k: float = None
    v: float = None
    u: float = None
    delta: float = None
    gama: float = None

    pNIU, pEIU, pNMSP, pEMSP, pNAMS, pEAMS = None, None, None, None, None, None
    N: torch.Tensor = None  # ITS Network Matrix
    LA = None
    LB = None
    X: float = None  # Probabilities related to dynamic deployment participation and exit mechanisms
    Y: float = None  # Probabilities related to dynamic deployment participation and exit mechanisms
    Xi: torch.Tensor = None  # Probabilities related to dynamic deployment participation and exit mechanisms at time t
    Yi: torch.Tensor = None  # Probabilities related to dynamic deployment participation and exit mechanisms at time t

    TH: float = None

    Kp: float = None
    Ki: float = None
    Kd: float = None
    beta: float = None
    ei: torch.Tensor = None
    Ii: torch.Tensor = None
    Di: torch.Tensor = None
    R1: torch.Tensor = None
    R2: torch.Tensor = None
    LambX: torch.Tensor = None
    LambY: torch.Tensor = None
    LambNIU: torch.Tensor = None
    LambEIU: torch.Tensor = None
    LambNMSP: torch.Tensor = None
    LambEMSP: torch.Tensor = None
    LambNAMS: torch.Tensor = None
    LambEAMS: torch.Tensor = None
    Lambf1: torch.Tensor = None
    Lambf2: torch.Tensor = None
    Lambf3: torch.Tensor = None
    Lambf4: torch.Tensor = None

    def __init__(self, Node: int, T: int):
        if Node <= 0:
            raise ValueError("Node count must be a positive integer.")
        try:
            self.LA = torch.zeros(Node, Node, dtype=torch.int)
            self.LB = torch.zeros(Node, Node, dtype=torch.int)
            self.pNIU = torch.zeros(Node, dtype=torch.float)
            self.pEIU = torch.zeros(Node, dtype=torch.float)
            self.pNMSP = torch.zeros(Node, dtype=torch.float)
            self.pEMSP = torch.zeros(Node, dtype=torch.float)
            self.pNAMS = torch.zeros(Node, dtype=torch.float)
            self.pEAMS = torch.zeros(Node, dtype=torch.float)
            self.T = T
            self.Xi = torch.zeros([T, Node], dtype=torch.float)
            self.Yi = torch.zeros([T, Node], dtype=torch.float)
            self.ei = torch.zeros(Node, dtype=torch.float)
            self.Ii = torch.zeros(Node, dtype=torch.float)
            self.Di = torch.zeros(Node, dtype=torch.float)
            self.LambX = torch.zeros(Node, dtype=torch.float)
            self.LambY = torch.zeros(Node, dtype=torch.float)
            self.LambNIU = torch.zeros(Node, dtype=torch.float)
            self.LambEIU = torch.zeros(Node, dtype=torch.float)
            self.LambNMSP = torch.zeros(Node, dtype=torch.float)
            self.LambEMSP = torch.zeros(Node, dtype=torch.float)
            self.LambNAMS = torch.zeros(Node, dtype=torch.float)
            self.LambEAMS = torch.zeros(Node, dtype=torch.float)
            self.Lambf1 = torch.zeros(Node, dtype=torch.float)
            self.Lambf2 = torch.zeros(Node, dtype=torch.float)
            self.Lambf3 = torch.zeros(Node, dtype=torch.float)
            self.Lambf4 = torch.zeros(Node, dtype=torch.float)
        except Exception as e:
            print("Network initialization failed:", e)

    def setBasicValue(self, delta, k, u, v, X, Y):
        self.delta = delta
        self.k = k
        self.u = u
        self.v = v
        self.X = X
        self.Y = Y

    def setP(self, Nodetype: str, Nodeid: int, p: float) -> None:
        # 确保所有必要的张量已初始化
        if any(attr is None for attr in [self.pNIU, self.pEIU, self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS]):
            raise ValueError("please initialize the network")

        # 确保 Nodeid 是有效的
        if Nodeid < 0:
            raise IndexError("Nodeid must be greater than 0.")

        # 创建节点类型与张量的映射字典
        node_dict = {
            "NIU": self.pNIU,
            "EIU": self.pEIU,
            "NMSP": self.pNMSP,
            "EMSP": self.pEMSP,
            "NAMS": self.pNAMS,
            "EAMS": self.pEAMS
        }

        # 检查 Nodetype 是否有效
        if Nodetype not in node_dict:
            raise ValueError("Node type not recognized.")

        tensor = node_dict[Nodetype]

        # 如果目标张量为 None，则抛出错误
        if tensor is None:
            raise ValueError(f"{Nodetype} has not been initialized.")

        # 检查 Nodeid 是否超出张量的范围
        if Nodeid > len(tensor):
            raise IndexError("Nodeid is out of bounds.")

        # 更新对应位置的值
        tensor[Nodeid] = p

    def connection_matrix(self, matrix_type: str, i: int, j: int):
        """Update the specified matrix with a value at position (i, j)."""
        if matrix_type == 'LA':
            self.LA[i, j] = 1
            self.LA[j, i] = 1
        elif matrix_type == 'LB':
            self.LB[i, j] = 1
            self.LB[j, i] = 1
        else:
            print("Invalid matrix type. Choose 'LA' or 'LB'.")

    def ITSNetwork_Connection_matrices(self):
        """Print the current state of matrices."""
        print("Adjacency Matrix LA:")
        print(self.LA)
        print("Adjacency Matrix LB:")
        print(self.LB)

    # set the value of Sita
    def setTheta(self, Theta1: float, Theta2: float, Theta3: float, Theta4: float) -> bool:
        try:
            self.theta1 = Theta1
            self.theta2 = Theta2
            self.theta3 = Theta3
            self.theta4 = Theta4
            return True
        except Exception as e:
            print("Sita initiates failed", e)
            return False

    # --------------f的计算
    def calculate_fi1(self, mij: torch.Tensor, p_i_EIU: torch.Tensor) -> torch.Tensor:
        """
        Calculate f_i^1 based on the formula:
        f_i^1 = ∏_j [1 - mij * p_i^EIU * θ1]
        """
        if mij.shape[0] != p_i_EIU.shape[0]:
            raise ValueError("Number of rows in mij must match the length of p_i_EIU.")

        effect = 1 - mij * p_i_EIU.unsqueeze(1) * self.theta1
        fi1 = torch.prod(effect, dim=1)
        return fi1

    def calculate_fi2(self, mij: torch.Tensor, p_i_EIU: torch.Tensor) -> torch.Tensor:
        """
        Calculate f_i^2 based on the formula:
        f_i^2 = ∏_j [1 - mij * p_i^EIU * θ2]
        """
        if mij.shape[0] != p_i_EIU.shape[0]:
            raise ValueError("Number of rows in mij must match the length of p_i_EIU.")

        effect = 1 - mij * p_i_EIU.unsqueeze(1) * self.theta2
        fi2 = torch.prod(effect, dim=1)
        return fi2

    def calculate_fi3(self, mij: torch.Tensor, p_i_EIU: torch.Tensor) -> torch.Tensor:
        """
        Calculate f_i^3 based on the formula:
        f_i^3 = ∏_j [1 - mij * p_i^EIU * θ3]
        """
        if mij.shape[0] != p_i_EIU.shape[0]:
            raise ValueError("Number of rows in mij must match the length of p_i_EIU.")

        effect = 1 - mij * p_i_EIU.unsqueeze(1) * self.theta3
        fi3 = torch.prod(effect, dim=1)
        return fi3

    def calculate_fi4(self, mij: torch.Tensor, p_i_EMSP: torch.Tensor) -> torch.Tensor:
        """
        Calculate f_i^4 based on the formula:
        f_i^4 = ∏_j [1 - mij * p_i^EMSP * θ4]
        """
        if mij.shape[0] != p_i_EMSP.shape[0]:
            raise ValueError("Number of rows in mij must match the length of p_i_EMSP.")

        effect = 1 - mij * p_i_EMSP.unsqueeze(1) * self.theta4
        fi4 = torch.prod(effect, dim=1)
        return fi4

    # IU的下一时刻的P的计算
    def calculate_IU_nexttime(self) -> None:
        if any(attr is None for attr in [self.pNIU, self.pEIU, self.theta1, self.delta]):
            raise ValueError("Please initialize pNIU, pEIU, sita1, and delta.")
        mij = self.LA  # Assuming LA matrix represents mij
        p_EIU = self.pEIU
        fi1 = self.calculate_fi1(mij, p_EIU)
        p_NIU_tNext = self.pNIU * fi1 + self.pEIU * self.delta
        p_EIU_tNext = self.pEIU * (1 - self.delta) + self.pNIU * (1 - fi1)
        self.pNIU, self.pEIU = p_NIU_tNext, p_EIU_tNext

    # MSP/AMS下一时刻的P的计算
    def calculate_MSP_AMS_nexttime(self) -> None:
        if any(attr is None for attr in
               [self.pNIU, self.pEIU, self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS, self.theta1, self.theta2,
                self.theta3, self.theta4, self.u, self.v, self.X, self.Y]):
            raise ValueError("Please initialize pNIU, pEIU, sita1, and delta.")
        mij = self.LA  # Assuming LA matrix represents mij
        p_EIU = self.pEIU
        p_EMSP = self.pEMSP
        fi2 = self.calculate_fi2(mij, p_EIU)
        fi3 = self.calculate_fi3(mij, p_EIU)
        fi4 = self.calculate_fi4(mij, p_EMSP)
        p_NMSP_tNext = self.pNMSP * fi2 * (
                1 - self.X) + self.pEMSP * self.v + self.pNAMS * fi3 + self.pEAMS * fi3 * fi4 * self.u + self.pNAMS * fi4 * self.Y
        p_EMSP_tNext = self.pEMSP * (1 - self.v) + self.pNMSP * (1 - fi2)
        p_NAMS_tNext = self.pNAMS * fi3 * fi4 * (1 - self.Y) + self.pNMSP * fi2 * self.X + self.pEAMS * self.u
        p_EAMS_tNext = self.pNAMS * (1 - fi3) + self.pNAMS * fi3 * (1 - fi4) + self.pEAMS * (1 - self.u)
        self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS = p_NMSP_tNext, p_EMSP_tNext, p_NAMS_tNext, p_EAMS_tNext

    def calculate_fki_thistime(self, k: int) -> torch.Tensor:
        sitas = [self.theta1, self.theta2, self.theta3, self.theta4]
        r_EIU = self.pEIU
        r_EMSP = self.pEMSP
        mij = self.LA
        if k in [1, 2, 3]:
            fkit = 1 - sitas[k - 1] * torch.sum(mij * r_EIU)
        elif k == 4:
            fkit = 1 - sitas[k - 1] * torch.sum(mij * r_EMSP)
        else:
            raise ValueError("k must be 1, 2, or 4")
        return fkit

    def calculate_Fk_thistime(self, k: int) -> torch.Tensor:
        sitas = [self.theta1, self.theta2, self.theta3, self.theta4]
        r_EIU = self.pEIU
        r_EMSP = self.pEMSP
        mij = self.LA
        if k in [1, 2, 3]:
            Fkt = sitas[k - 1] * torch.sum(mij * r_EIU)
        elif k == 4:
            Fkt = sitas[k - 1] * torch.sum(mij * r_EMSP)
        else:
            raise ValueError("k must be 1, 2, or 4")
        return Fkt

    def calculate_IU_nexttime_inF(self) -> None:
        if any(attr is None for attr in [self.pNIU, self.pEIU, self.theta1, self.delta]):
            raise ValueError("Please initialize pNIU, pEIU, sita1, and delta.")
        F1t = self.calculate_Fk_thistime(1)
        p_NIU_tNext = self.pNIU * (1 - F1t) + self.pEIU * self.delta
        p_EIU_tNext = self.pEIU * (1 - self.delta) + self.pNIU * F1t
        self.pNIU, self.pEIU = p_NIU_tNext, p_EIU_tNext

    def calculate_MSP_AMS_nexttime_inF(self) -> None:
        if any(attr is None for attr in
               [self.pNIU, self.pEIU, self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS, self.theta1, self.theta2,
                self.theta3, self.theta4, self.u, self.v, self.X, self.Y]):
            raise ValueError("Please initialize pNIU, pEIU, sita1, and delta.")
        F2t = self.calculate_Fk_thistime(2)
        F3t = self.calculate_Fk_thistime(3)
        F4t = self.calculate_Fk_thistime(4)
        p_NMSP_tNext = self.pNMSP * (1 - F2t) * (1 - self.X) + self.pEMSP * self.v + self.pNAMS * (1 - F3t) * (1 - F4t)
        p_EMSP_tNext = self.pEMSP * (1 - self.v) + self.pNMSP * F2t
        p_NAMS_tNext = self.pNAMS * (1 - F3t) * (1 - F4t) * (1 - self.Y) + self.pNMSP * self.u * (1 - F2t) + self.pEAMS
        p_EAMS_tNext = self.pNAMS * F3t + self.pNAMS * (1 - F3t) * F4t + self.pEAMS * (1 - self.u)
        self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS = p_NMSP_tNext, p_EMSP_tNext, p_NAMS_tNext, p_EAMS_tNext

    def generateMatrixN(self) -> None:
        self.k = self.pEMSP / self.pEIU
        self.gama = self.theta2 / self.theta1
        k = self.k
        gama = self.gama
        mij = self.LA
        self.N = 1 - (1 - gama) * self.pNMSP * mij

    def calculate_TH(self) -> None:
        if None in [self.N, self.delta, self.k, self.v]:
            raise ValueError("Please calculate N first.")
        else:
            delta = self.delta
            k = self.k
            v = self.v

            # Calculate the maximum eigenvalue of matrix self.N
            # Convert tensor to a float matrix for eigenvalue computation
            eigenvalues = torch.linalg.eigvals(self.N.float())
            phi_max_N = torch.max(torch.abs(eigenvalues))

            # Compute the stability threshold
            TH = (delta + k * v) / phi_max_N.item()
            self.TH = TH

    def calculate_Xi_Yi(self) -> None:
        eit = self.pEIU + self.pEAMS
        eit_1 = self.ei
        Iit_1 = self.Ii
        Iit = Iit_1 + eit
        Dit = eit - eit_1
        Xit_1 = self.Xi
        Yit_1 = self.Yi
        Kp = self.Kp
        Ki = self.Ki
        Kd = self.Kd
        Xit = Xit_1 + Kp * eit + Ki * Iit + Kd * Dit
        Yit = Yit_1 + Kp * eit + Ki * Iit + Kd * Dit
        self.ei = eit
        self.Ii = Iit
        self.Di = Dit
        self.Xi, self.Yi = Xit, Yit

    def calculate_NMSP_NAMS_nexttime(self) -> None:
        if any(attr is None for attr in
               [self.pNIU, self.pEIU, self.pNMSP, self.pEMSP, self.pNAMS, self.pEAMS, self.theta1, self.theta2,
                self.theta3, self.theta4, self.u, self.v, self.X, self.Y]):
            raise ValueError("Please initialize pNIU, pEIU, sita1, and delta.")
        mij = self.LA  # Assuming LA matrix represents mij
        p_EIU = self.pEIU
        p_EMSP = self.pEMSP
        fi2 = self.calculate_fi2(mij, p_EIU)
        fi3 = self.calculate_fi3(mij, p_EIU)
        fi4 = self.calculate_fi4(mij, p_EMSP)
        p_NMSP_tNext = self.pNMSP * fi2 * (
                1 - self.Xi) + self.pEMSP * self.v + self.pNAMS * fi3 + self.pEAMS * fi3 * fi4 * self.u + self.pNAMS * fi4 * self.Yi
        p_NAMS_tNext = self.pNAMS * fi3 * fi4 * (1 - self.Yi) + self.pNMSP * fi2 * self.Xi + self.pEAMS * self.u
        self.pNMSP, self.pNAMS = p_NMSP_tNext, p_NAMS_tNext

    def calculate_R1t_R2t(self) -> None:
        R1t = torch.sum(self.Xi, dim=0)
        R2t = torch.sum(self.Yi, dim=0)
        self.R1, self.R2 = R1t, R2t

    def calculate_w(self) -> torch.Tensor:
        beta = self.beta
        Xit = self.Xi
        Yit = self.Yi
        Ximin, lineXimin = torch.min(Xit, dim=0)
        Yimax, lineYimax = torch.max(Yit, dim=0)
        w = beta * torch.sum((torch.log(torch.relu(Xit - Ximin)) + torch.log(torch.relu(Yimax - Yit))), dim=(1, 0))
        return w

    def objective_function(self) -> torch.Tensor:
        pEIU = self.pEIU
        pEMSP = self.pEMSP
        LambX = self.LambX
        LambY = self.LambY
        Xi = self.Xi
        Yi = self.Yi
        w = self.calculate_w()
        self.calculate_R1t_R2t()
        R1t = self.R1
        R2t = self.R2
        obj = torch.sum(pEIU + pEMSP) + torch.sum(LambX * (torch.sum(Xi) - R1t) + LambY * (torch.sum(Yi) - R2t),
                                                  dim=1) + w
        return obj

    def calculate_X_Y_C(self) -> None:
        Xi = self.Xi
        Ximax = torch.max(Xi)
        Ximin = torch.min(Xi)
        Yi = self.Yi
        Yimax = torch.max(Yi)
        Yimin = torch.min(Yi)
        beta = self.beta
        LambNMSP = self.LambNMSP
        LambNAMS = self.LambNAMS
        LambX = self.LambX
        LambY = self.LambY
        p_NMSP = self.pNMSP
        p_EMSP = self.pEMSP
        p_NAMS = self.pNAMS
        mij = self.LA
        p_EIU = self.pEIU
        fi2 = self.calculate_fi2(mij, p_EIU)
        fi3 = self.calculate_fi3(mij, p_EIU)
        fi4 = self.calculate_fi4(mij, p_EMSP)
        A = (LambNMSP - LambNAMS) * p_NMSP * fi2 + LambX
        B = (LambNAMS - LambNMSP) * fi3 * fi4 * p_NAMS + LambY
        Xitmp1 = (Ximax + Ximin) / 2
        Xitmp2 = (Ximax - Ximin) / 2
        Yitmp1 = (Yimax + Yimin) / 2
        Yitmp2 = (Yimax - Yimin) / 2
        Xit = Xitmp1 + torch.sqrt(torch.pow(Xitmp2, 2) - beta / A)
        Yit = Yitmp1 + torch.sqrt(torch.pow(Yitmp2, 2) - beta / B)
        self.Xi, self.Yi = Xit, Yit

    def calculate_C(self) -> torch.Tensor:
        Xi = self.Xi
        Yi = self.Yi
        p = torch.cat((Xi, Yi), dim=1)
        C = torch.sum(p, dim=(1, 0))
        return C

    def PrintPs(self):
        print("pNIU", self.pNIU)
        print("pEIU", self.pEIU)
        print("pNAMS", self.pNAMS)
        print("pEAMS", self.pEAMS)
        print("pNMSP", self.pNMSP)
        print("pEMSP", self.pEMSP)

    def calculate_Next(self):
        self.calculate_IU_nexttime_inF()
        self.calculate_MSP_AMS_nexttime_inF()

    def calculateAndPrint_TH(self):
        self.generateMatrixN()
        self.calculate_TH()
        print(self.TH)

    # -------------------------------------------------

    def Lagrangian_function(self) -> torch.Tensor:
        Xi = self.Xi
        Yi = self.Yi
        R1 = self.R1
        R2 = self.R2
        v = self.v
        u = self.u
        delta = self.delta
        pNIU = self.pNIU
        pEIU = self.pEIU
        pNMSP = self.pNMSP
        pEMSP = self.pEMSP
        pNAMS = self.pNAMS
        pEAMS = self.pEAMS
        mij = self.LA
        fi1 = self.calculate_fi1(mij, pEIU)
        fi2 = self.calculate_fi2(mij, pEIU)
        fi3 = self.calculate_fi3(mij, pEIU)
        fi4 = self.calculate_fi4(mij, pEMSP)
        beta = self.beta
        LambNIU = self.LambNIU
        LambEIU = self.LambEIU
        LambNMSP = self.LambNMSP
        LambEMSP = self.LambEMSP
        LambNAMS = self.LambNAMS
        LambEAMS = self.LambEAMS
        Lambf1 = self.Lambf1
        Lambf2 = self.Lambf2
        Lambf3 = self.Lambf3
        Lambf4 = self.Lambf4
        LambX = self.LambX
        LambY = self.LambY
        sita1 = self.theta1
        sita2 = self.theta2
        sita3 = self.theta3
        sita4 = self.theta4
        part1 = torch.sum(pEIU + pEMSP, dim=0)
        part2 = torch.sum(LambX * torch.sum(Xi, dim=0) - R1, dim=0)
        part3 = beta * torch.sum(torch.log(Xi - torch.min(Xi, dim=0)) + torch.log(torch.max(Xi, dim=0) - Xi), dim=0)
        part4 = torch.sum(LambY * torch.sum(Yi, dim=0) - R2, dim=0)
        part5 = beta * torch.sum(torch.log(Yi - torch.min(Yi, dim=0)) + torch.log(torch.max(Yi, dim=0) - Yi), dim=0)
        part6 = torch.sum(LambNIU * (pNIU - fi1 - pEIU * delta), dim=(1, 0))
        part7 = torch.sum(LambEIU * (pEIU - delta * pEIU * (1 - delta) - pNIU * (1 - fi1)), dim=(1, 0))
        part8 = torch.sum(LambNMSP * (pNMSP - pNMSP * fi2 * (1 - Xi) - pEMSP - pNAMS * fi3 * fi4 * Yi), dim=(1, 0))
        part9 = torch.sum(LambEMSP * (pEMSP - pEMSP * (1 - v) + pNMSP * (1 - fi2)), dim=(1, 0))
        part10 = torch.sum(LambNAMS * (pNAMS - pNAMS * fi3 * fi4 * (1 - Yi) - pNMSP * fi2 * Xi - pEAMS * u), dim=(1, 0))
        part11 = torch.sum(LambEAMS * (pEAMS - pNAMS * (1 - fi3) - pNMSP * fi3 * Xi - pEAMS * u), dim=(1, 0))
        part12 = torch.sum(LambEAMS * (pEAMS - pNAMS * (1 - fi3) - pNAMS * fi3 * (1 - fi4) - pEAMS * (1 - u)),
                           dim=(1, 0))
        part13 = torch.sum(Lambf1 * (fi1 - torch.prod(1 - mij * pEIU * sita1, dim=0)), dim=(1, 0))
        part14 = torch.sum(Lambf2 * (fi2 - torch.prod(1 - mij * pEIU * sita2, dim=0)), dim=(1, 0))
        part15 = torch.sum(Lambf3 * (fi3 - torch.prod(1 - mij * pEIU * sita3, dim=0)), dim=(1, 0))
        part16 = torch.sum(Lambf4 * (fi4 - torch.prod(1 - mij * pEMSP * sita4, dim=0)), dim=(1, 0))
        L = part1 + part2 + part3 + part4 + part5 + part6 + part7 + part8 + part9 + part10 + part11 + part12 + part13 + part14 + part15 + part16
        return L

    def compute_pNIU(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pNIU = self.pNIU.grad
        return grad_pNIU

    def compute_pEIU(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pEIU = self.pEIU.grad
        return grad_pEIU

    def compute_pNMSP(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pNMSP = self.pNMSP.grad
        return grad_pNMSP

    def compute_pEMSP(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pEMSP = self.pEMSP.grad
        return grad_pEMSP

    def compute_pNAMS(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pNAMS = self.pNAMS.grad
        return grad_pNAMS

    def compute_pEAMS(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_pEAMS = self.pEAMS.grad
        return grad_pEAMS

    def compute_Xi(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_Xi = self.Xi.grad
        return grad_Xi

    def compute_Yi(self):
        L = self.Lagrangian_function()
        L.backward()
        grad_Yi = self.Yi.grad
        return grad_Yi

    def compute_fi1(self):
        mij = self.LA
        pEIU = self.pEIU
        fi1 = self.calculate_fi1(mij, pEIU)
        L = self.Lagrangian_function()
        L.backward()
        grad_fi1 = fi1.grad
        return grad_fi1

    def compute_fi2(self):
        mij = self.LA
        pEIU = self.pEIU
        fi2 = self.calculate_fi2(mij, pEIU)
        L = self.Lagrangian_function()
        L.backward()
        grad_fi2 = fi2.grad
        return grad_fi2

    def compute_fi3(self):
        mij = self.LA
        pEIU = self.pEIU
        fi3 = self.calculate_fi3(mij, pEIU)
        L = self.Lagrangian_function()
        L.backward()
        grad_fi3 = fi3.grad
        return grad_fi3

    def compute_fi4(self):
        mij = self.LA
        pEMSP = self.pEMSP
        fi4 = self.calculate_fi4(mij, pEMSP)
        L = self.Lagrangian_function()
        L.backward()
        grad_fi4 = fi4.grad
        return grad_fi4
