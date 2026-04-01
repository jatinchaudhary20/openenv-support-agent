from env.models import State, Action


class SupportEnv:
    def __init__(self):
        self.state = None

    def reset(self, task="easy"):
        if task == "easy":
            ticket = "Payment failed but money deducted"
        elif task == "medium":
            ticket = "Received wrong product, want refund"
        else:
            ticket = "App crashes when I open it"

        self.state = State(ticket=ticket)
        return self.state

    def step(self, action: Action):
        action_str = action.action_str.lower()
        reward = 0
        done = False

        if action_str.startswith("classify"):
            if "payment" in self.state.ticket.lower():
                if "billing" in action_str:
                    self.state.category = "billing"
                    reward = 1
            elif "refund" in self.state.ticket.lower() or "wrong" in self.state.ticket.lower():
                if "refund" in action_str:
                    self.state.category = "refund"
                    reward = 1
            else:
                if "technical" in action_str:
                    self.state.category = "technical"
                    reward = 1

        elif action_str.startswith("respond"):
            self.state.response = action_str
            reward = 1

        elif action_str.startswith("resolve"):
            if self.state.response:
                self.state.resolved = True
                reward = 1
                done = True
            else:
                reward = -1

        return self.state, reward, done, {}