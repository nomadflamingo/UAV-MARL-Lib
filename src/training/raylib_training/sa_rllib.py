import gymnasium as gym
import sys
import ray
from ray import tune
from ray.tune import Stopper
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    TRAINING_ITERATION_TIMER,
)
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.tune.registry import get_trainable_cls, register_env

sys.setrecursionlimit(3000)

# ——— CLI Argument Parsing ———
parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=100_000,
    default_reward=90.0,
)
parser.add_argument("--run", type=str, default="PPO", help="The RLlib algorithm to use.")
parser.add_argument("--env-name", type=str, default="quadx_waypoints")
args = parser.parse_args()

# ——— Optional Reward Wrapper (modular) ———
class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # Scale extreme rewards
        if reward >= 99.0 or reward <= -99.0:
            return reward / 10
        return reward

# ——— Custom Environment Creator ———
def create_quadx_waypoints_env(env_config):
    import PyFlyt.gym_envs  # Register envs
    from PyFlyt.gym_envs import FlattenWaypointEnv

    env = gym.make("PyFlyt/QuadX-Waypoints-v4")
    env = RewardWrapper(env)
    return FlattenWaypointEnv(env, context_length=1)

# ——— Custom Stopper for Training ———
class CustomStopper(Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id, result):
        return (
            result.get(TRAINING_ITERATION_TIMER, 0) >= args.stop_iters or
            result.get(f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", 0) >= args.stop_reward
        )

    def stop_all(self):
        return self.should_stop

# ——— Setup Ray and Register Env ———
ray.init(ignore_reinit_error=True, log_to_driver=True)
register_env(args.env_name, env_creator=create_quadx_waypoints_env)

# ——— Configure Algorithm ———
algo_cls = get_trainable_cls(args.run)
config = (
    algo_cls.get_default_config()
    .environment(env=args.env_name)
    .env_runners(num_envs_per_env_runner=4)
    .reporting(min_time_s_per_iteration=0.1)
)

if args.run == "PPO":
    config = config.rl_module(
        model_config={
            "fcnet_hiddens": [32],
            "fcnet_activation": "linear",
            "vf_share_layers": True,
        }
    ).training(
        minibatch_size=128,
        train_batch_size_per_learner=10000,
    )
elif args.run == "IMPALA":
    config = config.env_runners(num_env_runners=2)
    config = config.learners(num_gpus_per_learner=0)
    config = config.training(vf_loss_coeff=0.01)

# ——— Launch Training with Tuner ———
stop = CustomStopper()
try:
    tuner = tune.Tuner(
        args.run,
        run_config=ray.train.RunConfig(
            stop=stop,
            name="rllib_debug_run"
        ),
        param_space=config.to_dict(),
    )
    results = tuner.fit()

except Exception as e:
    import traceback
    print("⚠️ Exception during training:")
    traceback.print_exc()
