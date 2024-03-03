import numpy as np

class HMM:
    def __init__(self, states, observations, initial_prob, transition_prob, emission_prob):
        self.states = states
        self.observations = observations
        self.initial_prob = initial_prob
        self.transition_prob = transition_prob
        self.emission_prob = emission_prob

    def generate_sequence(self, length):
        sequence = []
        current_state = np.random.choice(len(self.states), p=self.initial_prob)
        for _ in range(length):
            observation = np.random.choice(len(self.observations), p=self.emission_prob[current_state])
            sequence.append(self.observations[observation])
            current_state = np.random.choice(len(self.states), p=self.transition_prob[current_state])
        return sequence

# Example
states = ['Sunny', 'Rainy']
observations = ['Dry', 'Damp', 'Wet']
initial_prob = np.array([0.8, 0.2])
transition_prob = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_prob = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])

hmm = HMM(states, observations, initial_prob, transition_prob, emission_prob)
generated_sequence = hmm.generate_sequence(10)
print("Generated Sequence:", generated_sequence)
