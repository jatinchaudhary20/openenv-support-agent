from env.support_env import SupportEnv
from env.models import Action

env = SupportEnv()

tasks = ["easy", "medium", "hard"]

for task in tasks:
    state = env.reset(task)
    total = 0

    actions = [
        "classify('billing')",
        "respond('ok')",
        "resolve()"
    ]

    for a in actions:
        state, reward, done, _ = env.step(Action(action_str=a))
        total += reward
        if done:
            break

    print(task, "score:", total)