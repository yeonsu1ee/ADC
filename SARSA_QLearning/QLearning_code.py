class QLearning:
    def __init__(self, actions, agent_indicator=10):
        self.actions = actions
        self.agent_indicator = agent_indicator
        self.alpha = 0.01
        self.gamma = 0.9
        self.epsilon = 0.2
        self.q_values = defaultdict(lambda: [0.0] * actions)
        
    #def _convert_state(self, s):
    #    return np.where(s == self.agent_indicator)[0][0]
    
    def _convert_state(self, obs):
        agent_position = np.argwhere(obs == self.agent_indicator)
        if agent_position.size > 0:
            return agent_position[0]
        return None

        
    def update(self, state, action, reward, next_state, next_action):
        state = self._convert_state(state)
        next_state = self._convert_state(next_state)
        
        q_value = self.q_values[state][action]

        ################## write code ################################
        next_q_value = self.q_values[next_state][next_action]
        
        td_error = reward + self.gamma * np.max(next_q_value - q_value)
        self.q_values[state][action] += self.alpha * td_error
        ##############################################################
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state = self._convert_state(state)
            q_values = self.q_values[state]
            action = np.argmax(q_values)
        return action
