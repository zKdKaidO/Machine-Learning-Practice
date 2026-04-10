import numpy as np

class HMM:
    def __init__(self, states, trans_mat, init_distribution, 
            emission_prob, observation):
        self.states = states
        self.trans_mat = trans_mat
        self.init_distribution = init_distribution
        self.emission_prob = emission_prob
        self.observation = observation
    
    def forward(self):
        # init
        n_states = len(self.states)
        n_observation = len(self.observation)
        alpha = np.zeros((n_observation, n_states), dtype=np.float64)

        # t=1
        for i in range(n_states):
            alpha[0][i] = self.init_distribution[i] * self.emission_prob[i][self.observation[0]]
        # t>1
        for i in range(1, n_observation):
            row_th = i
            for j in range(n_states):
                col_th = j
                coef = 0
                for k in range(n_states):
                    coef += alpha[row_th-1][k] * self.trans_mat[k][col_th] 
                alpha[row_th][col_th] = coef * self.emission_prob[col_th][self.observation[row_th]]
        return alpha, np.sum(alpha[n_observation-1], axis=1)

    def forecast(self, day: int):
        n_observation = len(self.observation)
        if day <= n_observation:
            return Exception("Nothing to forecast because you want the day already had observation!")
        beta = []
        alpha, _ = self.forward()
        for i in range(len(self.states)):
            beta.append(alpha[len(self.observation)-1][i])
        sum = np.sum(beta)
        normalized_beta = beta / sum
        
        forecast = np.dot(normalized_beta, np.linalg.matrix_power(self.trans_mat, day-n_observation))
        return forecast 

    def viterbi(self):
        n_states = len(self.states)
        n_observation = len(self.observation)
        v = np.zeros((n_observation, n_states), dtype=np.float64)
        ptr = np.zeros((n_observation, n_states), dtype=np.int64)

        for i in range(n_states):
            # t=1
            v[0][i] = self.init_distribution[i] * self.emission_prob[i][self.observation[0]]
            ptr[0][i] = 0

        for i in range(1, n_observation):
            row_th = i
            for j in range(n_states):
                col_th = j
                coef = (v[row_th-1] * self.trans_mat[:, col_th])
                v[row_th][col_th] = np.max(coef) * self.emission_prob[col_th][self.observation[row_th]]
                ptr[row_th][col_th] = np.argmax(coef)
        

        best_path = []
        for i in range(n_observation-1, -1, -1):
            if i == n_observation-1:
                best_col_last = np.argmax(v[i])
                best_path.append(best_col_last)
                continue
            best_col_last = ptr[i+1][best_col_last]
            best_path.append(best_col_last)
        best_path.reverse()

        return best_path

        
            
            




        






