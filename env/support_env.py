from env.models import State, Action

class SupportEnv:
    def __init__(self):
        self.state = None
        self.task = None
        self.expected_category = None

    def reset(self, task="easy"):
        self.task = task

        if task == "easy":
            self.state = State(ticket="Payment failed but money deducted")
            self.expected_category = "billing"

        elif task == "medium":
            self.state = State(ticket="I want refund for wrong product delivered")
            self.expected_category = "refund"

        elif task == "hard":
            self.state = State(ticket="App crashed after payment and I was charged twice")
            self.expected_category = "technical"

        return self.state

    def step(self, action: Action):
        action_str = action.action_str.lower()

        reward = 0
        done = False

        # CLASSIFY
        if "classify" in action_str:
            if self.state.category is not None:
                reward -= 1
            elif self.expected_category in action_str:
                reward += 1
                self.state.category = self.expected_category
            else:
                reward -= 0.5

        # RESPOND
        elif "respond" in action_str:
            if self.state.category is None:
                reward -= 1
            else:
                score = 0
                if "sorry" in action_str:
                    score += 0.3
                if "refund" in action_str:
                    score += 0.3
                if len(action_str) > 20:
                    score += 0.4

                reward += score
                self.state.response = action_str

        # RESOLVE
        elif "resolve" in action_str:
            if self.state.category and self.state.response:
                reward += 2
                self.state.resolved = True
                done = True
            else:
                reward -= 1

        # ESCALATE
        elif "escalate" in action_str:
            reward -= 0.5

        return self.state, reward, done, {}