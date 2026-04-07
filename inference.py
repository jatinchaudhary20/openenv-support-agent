from env.support_env import SupportEnv
from env.models import Action


def run_baseline():
    env = SupportEnv()
    tasks = ["easy", "medium", "hard"]

    for task in tasks:
        state = env.reset(task)
        total_reward = 0

        # START block
        print(f"[START] task={task}", flush=True)

        for step in range(1, 6):
            ticket = state.ticket.lower()

            # Simple deterministic policy
            if state.category is None:
                if "payment" in ticket:
                    action_str = "classify('billing')"
                elif "refund" in ticket or "wrong" in ticket:
                    action_str = "classify('refund')"
                else:
                    action_str = "classify('technical')"

            elif state.response is None:
                action_str = "respond('ok')"

            else:
                action_str = "resolve()"

            # Take step
            state, reward, done, _ = env.step(Action(action_str=action_str))
            total_reward += reward

            # STEP block
            print(f"[STEP] step={step} reward={reward}", flush=True)

            if done:
                break

        # END block
        print(f"[END] task={task} score={total_reward} steps={step}", flush=True)


if __name__ == "__main__":
    run_baseline()