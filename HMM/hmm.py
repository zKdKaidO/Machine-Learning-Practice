import numpy as np

class HMM:
    def __init__(self, states, trans_mat, init_distribution, 
            emission_prob, observation):
        self.states = states
        self.trans_mat = trans_mat
        self.init_distribution = init_distribution
        self.emission_prob = emission_prob
        self.observation = observation
    
    def forward(self, step):
        # init
        n_states = len(self.states)
        alpha = np.zeros((step, n_states), dtype=np.float64)

        # t=1
        for i in range(n_states):
            alpha[0][i] = self.init_distribution[i] * self.emission_prob[i][self.observation[0]]
        # t>1
        for i in range(1, step):
            row_th = i
            for j in range(n_states):
                col_th = j

                coef = 0
                for k in range(n_states):
                    coef += alpha[row_th-1][k] * self.trans_mat[k][col_th] 
                alpha[row_th][col_th] = coef * self.emission_prob[col_th][self.observation[row_th]]







