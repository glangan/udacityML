import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q_table = {}
        self.gamma = 0.3
        self.alpha = 0.9
        self.valid_states = [None, 'left', 'right', 'forward']
        self.traffic_light = ['red', 'green']
        self.epsilon = 0.1
        # initialize q-table
        for light in self.traffic_light:
            for way in self.valid_states:
                for l in self.valid_states:
                    for on in self.valid_states:
                        for action in self.valid_states:
                            self.q_table[(light, way, l, on), action] = 0.0

        self.penalties = 0
        self.total_successes = 0
        self.total_moves = 0
        self.net_reward = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.next_state = None
        self.net_reward = 0


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # TODO: Update state
        self.state = (inputs['light'], self.next_waypoint, inputs['left'], inputs['oncoming'])
        # print "Current state", self.state
        # TODO: Select action according to your policy
        action = self.get_action(self.state)
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward += reward
        if reward < 0:
            self.penalties += 1
        # TODO: Learn policy based on state, action, reward

        next_inputs = self.env.sense(self)
        self.next_state = (next_inputs['light'], self.planner.next_waypoint(), inputs['left'], inputs['oncoming'])
        max_q = self.get_max_q(self.next_state)

        self.q_table[self.state, action] = (1 - self.alpha) * self.q_table[self.state, action] \
                                                              + self.alpha * (reward + self.gamma * max_q)
        if reward == 12:
            self.total_successes += 1

        if reward == 12 or deadline == 0:
            print "Net reward", self.net_reward
        self.total_moves += 1

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_max_q(self, state):
        q_score = [self.q_table[state, a] for a in self.valid_states]
        return max(q_score)

    def get_action(self, state):
        q = {}
        for a in self.valid_states:
            q[a] = self.q_table[state, a]
        max_q = max(q.values())
        best_action = [k for k, v in q.items() if v == max_q]
        if len(best_action) > 1:
            action = random.choice(best_action)
        else:
            action = best_action[0]
        if random.random() < self.epsilon:
            action = random.choice(self.valid_states)
        return action


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track
    # Now simulate it
    sim = Simulator(e, update_delay=0.0001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit
    penalty_rate = a.penalties / float(a.total_moves)
    print "Total penalties: ", a.penalties
    print "Total successes: ", a.total_successes
    print "Penalty Rate: ", penalty_rate
    print "--------------------------------------------------"

if __name__ == '__main__':
    run()
