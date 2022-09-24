from stable_baselines3 import PPO
from SubGoalEnv import SubGoalEnv
# WIP"

def execute():
    episodes_per_task = 1
    titles = ["assembly-v2","basketball-v2","bin-picking-v2", "box-close-v2", "button-press-topdown-v2", "button-press-topdown-wall-v2", "button-press-v2", "button-press-wall-v2", "coffee-button-v2", "coffee-pull-v2", "coffee-push-v2", "dial-turn-v2", "disassemble-v2", "door-close-v2", "door-lock-v2", "door-open-v2", "door-unlock-v2", "hand-insert-v2", "drawer-close-v2", "drawer-open-v2", "faucet-open-v2", "faucet-close-v2", "hammer-v2", "handle-press-side-v2", "handle-press-v2","handle-pull-side-v2", "handle-pull-v2", "lever-pull-v2", "peg-insert-side-v2", "pick-place-wall-v2", "pick-out-of-hole-v2", "reach-v2", "push-back-v2", "push-v2", "pick-place-v2", "plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2", "plate-slide-back-side-v2", "peg-unplug-side-v2","soccer-v2", "stick-push-v2","stick-pull-v2", "push-wall-v2", "reach-wall-v2", "shelf-place-v2","sweep-into-v2","sweep-v2", "window-open-v2", "window-close-v2","Average"]
    all_tasks_mean_reward = 0
    all_tasks_mean_steps = 0
    all_tasks_success_rate = 0
    for i, title in enumerate(titles):
        env = make_env(title, "rew1", 50, i)()
        env = SubGoalEnv("pick-place-v2",render_subactions=True)
        print(env.observation_space)
        models_dir = "models/PPO"
        model_path = f"{models_dir}/3014656.zip"
        model = PPO.load(model_path, env=env)
        print("---------------------------")
        print(model.observation_space)
        mean_rew_all_tasks = 0
        num_success = 0
        mean_steps = 0
        for ep in range(episodes_per_task):
            obs = env.reset()
            done = False
            steps = 0
            total_reward = 0
            success = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                steps += 1
                total_reward += reward
                if info['success']:
                    success = True
                if done and success:
                    num_success += 1
            mean_rew_all_tasks += total_reward
            mean_steps += steps
        print('---------', title, ' ---------')
        print("mean_tot_rew:",mean_rew_all_tasks/episodes_per_task)
        print("mean_steps:", mean_steps/episodes_per_task)
        print("success rate:",num_success/episodes_per_task)
        all_tasks_mean_reward += mean_rew_all_tasks
        all_tasks_mean_steps += mean_steps
        all_tasks_success_rate += num_success
    print("\n-------all tasks:-------")
    print("mean_tot_rew:", all_tasks_mean_reward / (episodes_per_task*10))
    print("mean_steps:", all_tasks_mean_steps / (episodes_per_task*10))
    print("success rate:", all_tasks_success_rate / (episodes_per_task*10))


def make_env(name,rew_type,number_of_one_hot_tasks, one_hot_task_index):

    def _init():
        return SubGoalEnv(env=name, rew_type=rew_type, number_of_one_hot_tasks=number_of_one_hot_tasks,
                          one_hot_task_index=one_hot_task_index,render_subactions=False)
    return _init


if __name__ == '__main__':
    execute()