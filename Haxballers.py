# %% [markdown]
# ## Setup

# %%
# @title Import Libraries
import copy
import multiprocessing
import shutil
from typing import Any, Callable

import haxballgym
import numpy as np
import optuna
from haxballgym.utils.gamestates import GameState as HBGGameState
from haxballgym.utils.obs_builders import DefaultObs, ObsBuilder
from haxballgym.utils.reward_functions import CombinedReward, RewardFunction
from haxballgym.utils.state_setters import DefaultState, RandomState, StateSetter
from haxballgym.utils.terminal_conditions import TerminalCondition, common_conditions
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecMonitor
from ursinaxball import Game
from ursinaxball.common_values import ActionBin, BaseMap, GameState, TeamID
from ursinaxball.modules import Bot, GameScore, PlayerHandler

# %%
# @title Define Constants
# @markdown Sets up file paths for recordings, models, and logs.
from pathlib import Path
import warnings

FOLDER_REC = "recordings/"
FOLDER_MODELS = "models/"
FOLDER_TB_LOGS = "tensorboard_logs/"
FOLDER_OPTUNA_LOGS = "optuna_logs/"
FOLDER_MODELS_ID = "1NdFI8VP5vcev2Q7cSfVdgb0E827F4uvq"
FOLDER_TENSORBOARD_LOG_ID = "1l8JRQbgrx7JIQX9QWfNEUDIzU1BJ_R0K"
FOLDER_OPTUNA_LOGS_ID = "1R7qcpFuuT1LoIux0hVxaOFYywzudEa-r"
OPTUNA_STORAGE_FILE = "optuna_journal_storage.log"

Path(FOLDER_OPTUNA_LOGS).mkdir(parents=True, exist_ok=True)
Path(FOLDER_MODELS).mkdir(parents=True, exist_ok=True)
Path(FOLDER_TB_LOGS).mkdir(parents=True, exist_ok=True)

OPTUNA_STORAGE = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(  # pyright: ignore[reportAttributeAccessIssue]
        f"{FOLDER_OPTUNA_LOGS}/{OPTUNA_STORAGE_FILE}",
    )
)
TICK_SKIP = 15
N_ENVS = multiprocessing.cpu_count()

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [markdown]
# ## Define Bots
#
# Introduce classical bot policies used for scripted opponents and training benchmarks.

# %%
# @title Define Common Bots
# @markdown Initializes standard bots like RandomBot, ChaseBot, and GoalkeeperBot.
from ursinaxball.modules.bots import ChaseBot, GoalkeeperBot, RandomBot

BOTS: dict[str, type[Bot]] = {
    "RandomBot": RandomBot,
    "ChaseBot": ChaseBot,
    "GoalkeeperBot": GoalkeeperBot,
}

# %%
# @title Define Custom Bots
# @markdown Implements StrikerBot.
from ursinaxball.modules.bots.advanced_bots import follow_point, shoot_disc_close


class StrikerBot(Bot):
    """
    This bot chases the ball, positions itself behind it,
    and shoots towards the opposing goal.
    """

    def __init__(self, tick_skip: int = TICK_SKIP):
        super().__init__(tick_skip=tick_skip)
        self.previous_actions: list[int] = [0, 0, 0]

    def step_game(self, player: PlayerHandler, game: Game):
        ball = game.stadium_game.discs[0]

        # Determine the target goal
        target_goal = game.stadium_game.goals[1 if player.team == TeamID.RED else 0]

        # Calculate the center of the opposing goal
        goal_center = np.array(
            (
                (target_goal.points[0][0] + target_goal.points[1][0]) / 2,
                (target_goal.points[0][1] + target_goal.points[1][1]) / 2,
            )
        )

        # Set a target point slightly behind the ball
        vec_goal_to_ball = ball.position - goal_center
        offset = ball.radius + player.disc.radius
        strike_target = ball.position + (
            vec_goal_to_ball / np.linalg.norm(vec_goal_to_ball) * offset
        )

        # Generate movement and kick actions
        inputs_player = follow_point(player, strike_target.tolist(), 2)
        inputs_player[ActionBin.KICK] = shoot_disc_close(
            player, ball, int(player.disc.radius + ball.radius), self.previous_actions
        )

        self.previous_actions = inputs_player
        return inputs_player


BOTS["StrikerBot"] = StrikerBot

# %% [markdown]
# ## Configure Training Environment
#
# Configures the game, observations, rewards, terminal conditions, and environment wrappers used during training.

# %%
# @title Define Game
# @markdown Initializes the game instance with specific map settings.
import logging

DEFAULT_GAME = Game(
    stadium_file=BaseMap.CLASSIC,
    logging_level=logging.ERROR,
    enable_renderer=False,
    enable_recorder=False,
)

# %%
# @title Define Terminal Conditions
# @markdown Sets conditions for ending an episode, such as time limits or goals scored.

DEFAULT_TERMINAL_CONDITIONS: tuple[TerminalCondition, ...] = (
    common_conditions.TimeoutCondition(500),
    common_conditions.GoalScoredCondition(),
    common_conditions.NoTouchTimeoutCondition(50),
)

# %%
# @title Define Reward Function
# @markdown Configures the reward signal, combining event-based rewards and velocity-based shaping.
from haxballgym.utils.reward_functions.common_rewards import (
    AlignBallGoal,
    ConstantReward,
    EventReward,
)
from haxballgym.utils.reward_functions.velocity_reward import VelocityBallToGoalReward
from ursinaxball.objects import Stadium


class NewAlignBallGoal(AlignBallGoal):
    def __init__(self, stadium: Stadium, defense=1.0, offense=1.0, gamma=0.99):
        super().__init__(stadium, defense, offense)
        self.gamma = gamma
        self.prev_alignment: dict[str, float] = {}

    def reset(self, initial_state: HBGGameState):
        super().reset(initial_state)
        self.prev_alignment.clear()

    def get_reward(
        self, player: PlayerHandler, state: HBGGameState, previous_action: np.ndarray
    ) -> float:
        reward = super().get_reward(player, state, previous_action)
        prev_alignment = self.prev_alignment.get(player.name, reward)
        self.prev_alignment[player.name] = reward
        return self.gamma * reward - prev_alignment


class TeamReward(RewardFunction):
    """
    Takes an individual stateless reward and averages it across the entire team.
    """

    def __init__(self, reward_fn: RewardFunction):
        super().__init__()
        self.reward_fn = reward_fn

    def reset(self, initial_state: HBGGameState):
        self.reward_fn.reset(initial_state)

    def get_reward(
        self, player: PlayerHandler, state: HBGGameState, previous_action: np.ndarray
    ) -> float:
        teammates = [p for p in state.players if p.team == player.team]
        return sum(
            self.reward_fn.get_reward(p, state, previous_action) for p in teammates
        ) / len(teammates)


DEFAULT_REWARD_FN = CombinedReward(
    (
        ConstantReward(),
        EventReward(
            team_goal=10.0,
            team_concede=-10.0,
            touch=0.04,
            kick=0.04,
        ),
        VelocityBallToGoalReward(DEFAULT_GAME.stadium_game),
    ),
    reward_weights=(-0.001, 1.0, 0.02),
)

# %%
# @title Define Observation Builder
# @markdown Creates `EgocentricObs` to normalize and center observations around the agent.
import numpy.typing as npt


class EgocentricObs(DefaultObs):
    """
    Builds an egocentric observation space for the agent.
    All positions (ball, allies, enemies) are normalized relative to the agent's current position. Missing allies and enemies are padded to be off-screen.

    Observation Array Shape (Length: 11 + 4 * (max_allies + max_enemies)):
    Indices:
    [0:4]   - ball.x, ball.y, ball.vx, ball.vy (relative to self)
    [4]     - game.state (0: Kickoff, 1: Playing)
    [5]     - game.team_kickoff (1: Own team kicking off, 0: Other team kicking off)
    [6]     - self.is_kicking (1: True, 0: False)
    [7:11]  - self.x, self.y, self.vx, self.vy (absolute normalized positions)

    Dynamic Indices (Based on team sizes):
    [11:...] - (ally.x, ally.y, ally.vx, ally.vy) for ally in allies (relative to self)
    [...:...] - (enemy.x, enemy.y, enemy.vx, enemy.vy) for enemy in enemies (relative to self)
    """

    MAX_VEL = 15.0

    def __init__(self, max_allies=2, max_enemies=3):
        super().__init__()
        self.max_allies = max_allies
        self.obs_len = 11 + 4 * max_allies + 4 * max_enemies

    def build_obs(
        self,
        player: PlayerHandler,
        state: HBGGameState,
        previous_action: npt.NDArray[np.int_],
    ):
        obs: npt.NDArray[np.float64] = super().build_obs(player, state, previous_action)
        final_obs = np.full(self.obs_len, 4.0)

        width = state.stadium_game.width
        height = state.stadium_game.height

        self_x, self_y = obs[7] / width, obs[8] / height

        def norm(mvt: npt.NDArray[np.float64]):
            return mvt / np.array(
                [width, height, EgocentricObs.MAX_VEL, EgocentricObs.MAX_VEL]
            ) - np.array([self_x, self_y, 0, 0])

        final_obs[0:4] = norm(obs[0:4])
        for i in range(7, 7 + 4 * len(state.players), 4):
            final_obs[i : i + 4] = norm(obs[i : i + 4])
        final_obs[7] += self_x
        final_obs[8] += self_y

        final_obs[4] = state.state  # Kickoff / Playing
        final_obs[5] = player.team == state.team_kickoff
        final_obs[6] = player.is_kicking()

        # Sort allies and enemies by distance to ball
        allies = len([p for p in state.players if p.team == player.team]) - 1
        if allies > 0:
            allies_obs = [obs[i : i + 4] for i in range(11, 11 + 4 * allies, 4)]
            allies_obs.sort(key=lambda x: np.linalg.norm(x[0:2] - obs[0:2]))
            final_obs[11 : 11 + 4 * allies] = np.concatenate(allies_obs)

        if (enemies := len(state.players) - allies - 1) > 0:
            enemies_obs = [
                obs[i : i + 4]
                for i in range(11 + 4 * allies, 11 + 4 * allies + 4 * enemies, 4)
            ]
            enemies_obs.sort(key=lambda x: np.linalg.norm(x[0:2] - obs[0:2]))
            final_obs[
                11 + 4 * self.max_allies : 11 + 4 * self.max_allies + 4 * enemies
            ] = np.concatenate(enemies_obs)

        return final_obs


DEFAULT_OBS_BUILDER = EgocentricObs(max_allies=2, max_enemies=3)


# %%
# @title Define State Setter
# @markdown Defines state setters to initialize the environment in specific scenarios like kickoffs or drills.
class CombinedState(StateSetter):
    def __init__(
        self,
        state_setters: tuple[StateSetter, ...],
        state_weights: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.state_setters = state_setters
        if state_weights is None:
            self.state_weights = np.ones(len(state_setters), dtype=float)
        else:
            self.state_weights = np.array(state_weights, dtype=float)
        self.state_weights /= self.state_weights.sum()
        self._rng = np.random.default_rng()

    def reset(self, game: Game, save_recording: bool):
        idx = self._rng.choice(len(self.state_weights), p=self.state_weights)
        self.state_setters[idx].reset(game, save_recording)


class RandomKickoff(DefaultState):
    def __init__(self, red_percent=0.5):
        super().__init__()
        self.red_percent = red_percent
        self._rng = np.random.default_rng()

    def reset(self, game: Game, save_recording: bool):
        super().reset(game, save_recording)
        game.state = GameState.KICKOFF
        game.team_kickoff = (
            TeamID.RED if self._rng.random() < self.red_percent else TeamID.BLUE
        )


class StrikerDrill(RandomState):
    def reset(self, game: Game, save_recording: bool):
        super().reset(game, save_recording)

        ball, *placed = game.stadium_game.discs
        placed.extend(player.disc for player in game.players)
        if (game.team_kickoff == TeamID.RED) == (ball.position[0] < 0):
            ball.position[0] *= -1
            for player in game.players:
                player.disc.position[0] *= -1
        ball.velocity /= 5

        for player in game.players:
            if player.team == game.team_kickoff:
                pos = player.disc.position
                radius = player.disc.radius
                offset = np.array(
                    [
                        ball.radius + radius + self._rng.uniform(0, radius),
                        self._rng.uniform(-2 * radius, 2 * radius),
                    ],
                    dtype=float,
                )
                offset[0] *= -1 if player.team == TeamID.RED else 1
                player.disc.position = ball.position + offset
                if not self.is_valid_position(player.disc, placed):
                    player.disc.position = pos
                break


class GoalkeeperDrill(RandomState):
    def reset(self, game: Game, save_recording: bool):
        super().reset(game, save_recording)

        ball, *placed = game.stadium_game.discs
        placed.extend(player.disc for player in game.players)
        if (game.team_kickoff == TeamID.RED) == (ball.position[0] > 0):
            ball.position[0] *= -1
            for player in game.players:
                player.disc.position[0] *= -1

        # Determine the team goal
        goal = (
            game.stadium_game.goals[0]
            if game.team_kickoff == TeamID.RED
            else game.stadium_game.goals[1]
        )
        # Calculate the target in the team goal
        goal_target = goal.points[0] + self._rng.random() * (
            goal.points[1] - goal.points[0]
        )

        vec_ball_to_goal = goal_target - ball.position
        vec_norm = np.linalg.norm(vec_ball_to_goal)
        ball.velocity = vec_ball_to_goal / vec_norm * 5

        for player in game.players:
            if player.team == game.team_kickoff:
                pos = player.disc.position

                # Pick a random point on the ball->goal line, biased toward the goal.
                anchor = ball.position + vec_ball_to_goal * (
                    0.55 + 0.4 * np.sqrt(self._rng.random())
                )

                # Move orthogonally from that anchor with bounded offset.
                ortho = np.array([-vec_ball_to_goal[1], vec_ball_to_goal[0]])
                ortho /= np.linalg.norm(ortho)
                lateral_offset = (
                    -1.0 if self._rng.random() < 0.5 else 1.0
                ) * self._rng.uniform(
                    ball.radius + player.disc.radius,
                    ball.radius + 3 * player.disc.radius,
                )

                player.disc.position = anchor + ortho * lateral_offset
                if not self.is_valid_position(player.disc, placed):
                    player.disc.position = pos
                break


class PassingDrill(RandomState):
    pass


DEFAULT_STATE_SETTER = CombinedState(
    (
        RandomKickoff(red_percent=0.8),
        RandomState(),
        StrikerDrill(),
    ),
    state_weights=(
        0.6,
        0.2,
        0.2,
    ),
)

# %%
# @title Define Haxball Parallel Environment
# @markdown Adapts the Haxball environment for multi-agent reinforcement learning using PettingZoo.
from pettingzoo import ParallelEnv


class HaxballParallelEnv(ParallelEnv):
    """
    Wraps the standard HaxballGym environment into a PettingZoo Parallel Environment for multi-agent reinforcement learning.
    """

    metadata = {"render_modes": [], "name": "haxball_pe_v1"}

    def __init__(self, team_size: int, bots: list[Bot] | None, **make_kwargs):
        self.render_mode = None
        self.gym_env = haxballgym.make(team_size=team_size, bots=bots, **make_kwargs)

        self.possible_agents: list[str] = [f"red_{i}" for i in range(team_size)]
        if bots is None:
            self.possible_agents += [f"blue_{i}" for i in range(team_size)]

    def step(self, actions: dict):  # pyright: ignore[reportIncompatibleMethodOverride]
        obs, reward, terminated, truncated, info = self.gym_env.step(
            list(actions.values())
        )
        if len(self.agents) == 1:
            obs = [obs]
            reward = [reward]

        observations = dict(zip(self.agents, obs))
        rewards = dict(zip(self.agents, reward))
        terminations = dict.fromkeys(self.agents, terminated)
        truncations = dict.fromkeys(self.agents, truncated)
        infos = dict.fromkeys(self.agents, info)

        if terminated or truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        obs, info = self.gym_env.reset(seed=seed, options=options)
        if len(self.agents) == 1:
            obs = [obs]

        observations = dict(zip(self.agents, obs))
        infos = dict.fromkeys(self.agents, info)

        return observations, infos

    def observation_space(self, agent):
        return self.gym_env.observation_space

    def action_space(self, agent):
        return self.gym_env.action_space

    def close(self):
        return self.gym_env.close()


# %%
# @title Define Make Environment
# @markdown Factory function to create vectorized environments for training with Stable-Baselines3.
import supersuit as ss
from supersuit.vector.concat_vec_env import ConcatVecEnv


if not hasattr(ConcatVecEnv, "seed"):

    def seed_dummy(self, seed=None):
        return [None] * self.num_envs

    ConcatVecEnv.seed = seed_dummy  # pyright: ignore[reportAttributeAccessIssue]


def make_env(
    game: Game = DEFAULT_GAME,
    tick_skip=TICK_SKIP,
    team_size=1,
    terminal_conditions: tuple[TerminalCondition, ...] = DEFAULT_TERMINAL_CONDITIONS,
    bots: list[Bot] | None = [],
    reward_fn: RewardFunction = DEFAULT_REWARD_FN,
    obs_builder: ObsBuilder = DEFAULT_OBS_BUILDER,
    state_setter: StateSetter = DEFAULT_STATE_SETTER,
):
    env = HaxballParallelEnv(
        game=copy.deepcopy(game),
        tick_skip=tick_skip,
        team_size=team_size,
        bots=bots,
        reward_fn=reward_fn,
        terminal_conditions=terminal_conditions,
        obs_builder=obs_builder,
        state_setter=state_setter,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=N_ENVS, base_class="stable_baselines3"
    )

    return VecMonitor(env)


# %% [markdown]
# ## Sync Data
# Load and save model checkpoints, TensorBoard logs, and Optuna logs from Google Drive.

# %%
# @title Define Drive
# @markdown Authenticates and mounts Google Drive for saving/loading models and logs.
from pathlib import Path

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pydrive2.files import GoogleDriveFile

try:
    from google.colab import auth  # pyright: ignore[reportMissingImports]
    from oauth2client.client import (  # pyright: ignore[reportMissingImports]
        GoogleCredentials,
    )

    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
except:
    if Path("client_secrets.json").exists():
        with open("client_secrets.json", "r") as f:
            client_secrets = f.read()
    else:
        with open("client_secrets.json", "w") as f:
            f.write(input("client_secrets.json: "))
    gauth = GoogleAuth()
    gauth.settings["get_refresh_token"] = True
    gauth.CommandLineAuth()

drive = GoogleDrive(gauth)  # pyright: ignore[reportPossiblyUnboundVariable]


# %%
# @title Define ModelBot Wrapper
# @markdown Wraps a trained model to function as a bot within the game logic.
def ModelBot(
    model_path: str,
    obs_builder: ObsBuilder = DEFAULT_OBS_BUILDER,
):
    """
    Wraps a trained Stable-Baselines3 model so it can be used as a standard Bot in the game.
    Handles padding or truncating the observation array if the bot is placed in a match with a different number of allies/enemies than it was trained for.
    """

    class _ModelBot(Bot):
        def __init__(self, tick_skip=TICK_SKIP):
            super().__init__(tick_skip=tick_skip)
            self.model = PPO.load(model_path)

        def step_game(self, player: PlayerHandler, game: Game):
            obs = obs_builder.build_obs(
                player, HBGGameState(game), np.zeros(3, dtype=float)
            )
            action, _states = self.model.predict(obs, deterministic=True)
            action -= np.array([1, 1, 0], dtype=int)
            action[0] *= 1 if player.team == TeamID.RED else -1
            return action.tolist()

    return _ModelBot


# %% [markdown]
# ### Load Data
# Load model checkpoints, tensorboard logs, and optuna logs from Google Drive


# %%
# @title Load Model Checkpoints
# @markdown Load model checkpoints from Google Drive.
# @title Load Model Checkpoints
def load_model(model_file: GoogleDriveFile, obs_builder=DEFAULT_OBS_BUILDER):
    file_name = model_file.get("title") or ""
    model_path = f"{FOLDER_MODELS}/{file_name}"
    model_file.GetContentFile(model_path)
    bot = ModelBot(model_path, obs_builder)
    BOTS[file_name.rstrip(".zip")] = bot
    print(f"Loaded {file_name}")
    return bot


MODELS: dict[str, GoogleDriveFile] = {
    model["title"]: model
    for model in drive.ListFile(
        {
            "q": f"'{FOLDER_MODELS_ID}' in parents and mimeType='application/zip' and trashed=false"
        }
    ).GetList()
}
for model_file in MODELS.values():
    load_model(model_file)


# %%
# @title Load Tensorboard Logs
# @markdown Load TensorBoard logs from Google Drive.
def load_tb_log(log_file: GoogleDriveFile):
    file_name = log_file.get("title") or ""
    log_path = f"{FOLDER_TB_LOGS}/{file_name}"
    log_file.GetContentFile(log_path)
    shutil.unpack_archive(log_path, FOLDER_TB_LOGS)
    print(f"Loaded Tensorboard Log {file_name}")


TB_LOGS: dict[str, GoogleDriveFile] = {
    log["title"]: log
    for log in drive.ListFile(
        {
            "q": f"'{FOLDER_TENSORBOARD_LOG_ID}' in parents and mimeType='application/zip' and trashed=false"
        }
    ).GetList()
}
for tb_log_file in TB_LOGS.values():
    load_tb_log(tb_log_file)


# %%
# @title Load Optuna Logs
# @markdown Load Optuna logs from Google Drive.
def load_optuna_log(optuna_log_file: GoogleDriveFile):
    file_name = optuna_log_file.get("title") or ""
    optuna_log_file.GetContentFile(f"{FOLDER_OPTUNA_LOGS}/{file_name}")
    print(f"Loaded Optuna Log {file_name}")


OPTUNA_LOGS: dict[str, GoogleDriveFile] = {
    log["title"]: log
    for log in drive.ListFile(
        {"q": f"'{FOLDER_OPTUNA_LOGS_ID}' in parents and trashed=false"}
    ).GetList()
}
for optuna_log_file in OPTUNA_LOGS.values():
    load_optuna_log(optuna_log_file)

# %% [markdown]
# ### Save Data
# Save model checkpoints, tensorboard logs, and optuna logs to Google Drive.


# %%
# @title Save Model Checkpoints
# @markdown Save model checkpoints to Google Drive.
def save_model_path(model_path: str):
    model_name = Path(model_path).name
    if model_name in MODELS:
        model_file = MODELS[model_name]
    else:
        model_file = drive.CreateFile(
            {"title": model_name, "parents": [{"id": FOLDER_MODELS_ID}]}
        )
    model_file.SetContentFile(model_path)
    try:
        model_file.Upload()
        MODELS[model_name] = model_file
        print(f"Saved Model {model_path}")
    except Exception as e:
        print(f"Fail Save Model {model_path}: {e}")
    return model_file


def save_model(model: BaseAlgorithm, model_name: str):
    model_path = f"{FOLDER_MODELS}/{model_name}.zip"
    model.save(model_path)
    BOTS[model_name] = ModelBot(model_path)
    return save_model_path(model_path)


# %%
# @title Save Tensorboard Logs
# @markdown Save TensorBoard logs to Google Drive.
def save_tb_logs_path(model_name: str):
    # model_name looks like "ppo-3v3_LeaguePlay-c_v1_1"
    zip_file_name = f"{model_name}.zip"
    zip_path = f"{FOLDER_TB_LOGS}/{model_name}"

    try:
        shutil.make_archive(
            zip_path, "zip", root_dir=FOLDER_TB_LOGS, base_dir=model_name
        )

        if zip_file_name in TB_LOGS:
            tb_log_file = TB_LOGS[zip_file_name]
        else:
            tb_log_file = drive.CreateFile(
                {"title": zip_file_name, "parents": [{"id": FOLDER_TENSORBOARD_LOG_ID}]}
            )
            TB_LOGS[zip_file_name] = tb_log_file

        tb_log_file.SetContentFile(f"{zip_path}.zip")
        tb_log_file.Upload()
        print(f"Saved Tensorboard Log {zip_file_name}")
        return tb_log_file

    except Exception as e:
        print(f"Failed to save Tensorboard Log {model_name}: {e}")
        return None


def save_tb_logs(model: BaseAlgorithm):
    log_path = Path(str(model.logger.dir))
    return save_tb_logs_path(log_path.name)


# %%
# @title Save Optuna Logs
# @markdown Save Optuna logs to Google Drive
def save_optuna_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """Callback to save the Optuna journal to Google Drive after each trial."""
    if OPTUNA_STORAGE_FILE in OPTUNA_LOGS:
        optuna_log = OPTUNA_LOGS[OPTUNA_STORAGE_FILE]
    else:
        optuna_log = drive.CreateFile(
            {"title": OPTUNA_STORAGE_FILE, "parents": [{"id": FOLDER_OPTUNA_LOGS_ID}]}
        )
    optuna_log.SetContentFile(
        OPTUNA_STORAGE._backend._file_path  # pyright: ignore[reportAttributeAccessIssue]
    )
    try:
        optuna_log.Upload()
        print(f"Saved Optuna Log {OPTUNA_STORAGE_FILE}")
    except Exception as e:
        print(f"Failed to save Optuna journal to Drive: {e}")


# %% [markdown]
# ## Define Training
#
# Defines training utilities, checkpoint callbacks, and curriculum training workflows.

# %%
# @title Define Train Model
# @markdown Configures the training loop with checkpoint callbacks to save progress.
from stable_baselines3.common.callbacks import CheckpointCallback


class SaveModelCheckpointCallback(CheckpointCallback):
    def _on_step(self):
        continue_training = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            save_model_path(model_path)
            save_tb_logs(self.model)
        return continue_training


def train_model(
    model: BaseAlgorithm,
    model_name: str,
    total_timesteps: int,
    save_freq: int = 65_536,
    progress_bar: bool = True,
):
    model._setup_model()

    checkpoint_callback = SaveModelCheckpointCallback(
        save_freq=save_freq,
        save_path=f"{FOLDER_MODELS}/{model_name}",
        name_prefix=model_name,
    )
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=model_name,
        reset_num_timesteps=False,
        progress_bar=progress_bar,
        callback=checkpoint_callback,
    )
    model.env.close()  # pyright: ignore[reportOptionalMemberAccess]
    save_model(model, model_name)
    save_tb_logs(model)
    return model


# %%
# @title Define Curriculum
# @markdown Implements curriculum learning by training sequentially on different environment configurations.
def train_curriculum(
    model: BaseAlgorithm,
    model_name: str,
    envs: list[VecMonitor],
    hyperparams: list[dict[str, Any]],
    total_timesteps: list[int],
    save_freq: int = 65_536,
    progress_bar: bool = True,
):
    for i in range(len(envs)):
        model.__dict__.update(hyperparams[i])
        model.set_env(envs[i])
        train_model(
            model,
            model_name=f"{model_name}_drill-{i}",
            total_timesteps=total_timesteps[i],
            save_freq=save_freq,
            progress_bar=progress_bar,
        )
    save_model(model, model_name)
    return model


# %%
# @title Define Optimize Hyperparameters
# @markdown Runs the Optuna study using a generic optimization function.
from stable_baselines3.common.evaluation import evaluate_policy


def suggest_ppo_hyperparams(trial: optuna.Trial):
    """Learning hyperparameters we want to optimize"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
    }


def optimize_hyperparams(
    model: BaseAlgorithm,
    suggest_hyperparams: Callable[
        [optuna.Trial], dict[str, Any]
    ] = suggest_ppo_hyperparams,
    study_name: str | None = None,
    n_trials: int = 20,
    total_timesteps: int = 100_000,
    show_progress_bar=True,
):
    def objective(trial: optuna.Trial):
        model.__dict__.update(  # pyright: ignore[reportAttributeAccessIssue]
            suggest_hyperparams(trial)
        )
        model._setup_model()

        model.learn(total_timesteps=total_timesteps)

        mean_reward, _ = evaluate_policy(
            model,
            model.get_env(),  # pyright: ignore[reportArgumentType]
            n_eval_episodes=20,
            deterministic=True,
        )

        return mean_reward

    study = optuna.create_study(
        storage=OPTUNA_STORAGE,
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        callbacks=[save_optuna_callback],
    )
    print("Best reward:", study.best_value)
    print("Best params:", study.best_params)
    return study


DEFAULT_PPO_HYPERPARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "tensorboard_log": FOLDER_TB_LOGS,
    "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
    "device": "cpu",
}

# %% [markdown]
# ## Define League Training
#

# %% [markdown]
# # Optimize Hyperparameters

# %%
# @title Start Optuna Dashboard
load_hpo = False  # @param {type:"boolean","placeholder":"True"}
if load_hpo:
    try:
        from google.colab import output  # pyright: ignore[reportMissingImports]
        from optuna_dashboard import run_server
        import threading

        dashboard_thread = threading.Thread(
            target=lambda: run_server(OPTUNA_STORAGE, port=8787)
        )
        dashboard_thread.start()
        output.serve_kernel_port_as_iframe(8787, path="/dashboard", height="1200")
    except:
        print("Optuna Dashboard is only supported in Google Colab.")

# %%
# @title Optimize Hyperparameters for 1v1 Goalkeeper
study_name = "1v1_Striker_v1"
if load_hpo:
    if study_name in optuna.get_all_study_names(storage=OPTUNA_STORAGE):
        DEFAULT_PPO_HYPERPARAMS.update(
            optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE).best_params
        )
    else:
        model = PPO(
            **DEFAULT_PPO_HYPERPARAMS,
            env=make_env(
                team_size=1,
                bots=[BOTS["StrikerBot"](tick_skip=TICK_SKIP)],
                state_setter=DEFAULT_STATE_SETTER,
            ),
        )
        study = optimize_hyperparams(
            model,
            suggest_ppo_hyperparams,
            study_name=study_name,
            n_trials=10,
            total_timesteps=500_000,
        )
        DEFAULT_PPO_HYPERPARAMS.update(study.best_params)
DEFAULT_PPO_HYPERPARAMS

# %% [markdown]
# # Train Models
#
# Launches training runs, logs progress with TensorBoard, and saves checkpoints for later evaluation.
#
# [models - Google Drive](https://drive.google.com/drive/folders/1NdFI8VP5vcev2Q7cSfVdgb0E827F4uvq)
#
# Bypass Colab Timeout:
# - Run in console:
# ```js
# setInterval(document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click,30000)
# ```
# - Save bookmark:
# ```
# javascript:(setInterval(document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click,30000))
# ```
#

# %% [markdown]
# ## Curriculum Training

# %% [markdown]
# ### Train PPO 1v0
# Trains an agent in a 1v0 striker drill and a 1v0 random kickoff drill.

# %%
# @title Train PPO 1v0 Striker Drill
# @markdown Trains an agent in a 1v0 striker drill.
model_name = "ppo-1v0_StrikerDrill_v0"
env = make_env(
    team_size=1,
    bots=[],
    state_setter=StrikerDrill(red_percent=1.0),
)
if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    model = PPO(
        **DEFAULT_PPO_HYPERPARAMS,
        env=env,
    )
    model = train_model(
        model,
        model_name=model_name,
        total_timesteps=1_000_000,
    )

# %% [markdown]
# ### Train PPO 1v1
# Trains an agent in a 1v1 striker drill, a 1v1 goalkeeper drill, and a 1v1 self-play scenario.

# %%
# @title Train PPO 1v1 against StrikerBot
# @markdown Trains an agent in a 1v1 against a StrikerBot.
model_name = "ppo-1v1_vs-Striker_v1"
env = make_env(
    team_size=1,
    bots=[BOTS["StrikerBot"](tick_skip=TICK_SKIP)],
    state_setter=DEFAULT_STATE_SETTER,
)
if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    new_model = PPO(
        **DEFAULT_PPO_HYPERPARAMS,
        env=env,
    )
    new_model.set_parameters(
        model.get_parameters()  # pyright: ignore[reportArgumentType]
    )
    model = train_model(
        new_model,
        model_name=model_name,
        total_timesteps=2_000_000,
    )

# %%
# @title Train PPO 1v1 Self-Play
# @markdown Trains an agent in a 1v1 self-play scenario.
model_name = "ppo-1v1_Self-Play_v2"
env = make_env(
    team_size=1,
    bots=None,
    state_setter=DEFAULT_STATE_SETTER,
)
if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    new_model = PPO(
        **DEFAULT_PPO_HYPERPARAMS,
        env=env,
    )
    new_model.set_parameters(
        model.get_parameters()  # pyright: ignore[reportArgumentType]
    )
    model = train_model(
        new_model,
        model_name=model_name,
        total_timesteps=2_000_000,
        # progress_bar=False,
    )

# %% [markdown]
# ### Train PPO 2v2
# Trains an agent in a 2v2 self-play scenario.

# %%
# @title Train PPO 2v2 Self-Play
# @markdown Trains an agent in a 2v2 self-play scenario.
model_name = "ppo-2v2_Self-Play_v2"
env = make_env(
    team_size=2,
    bots=None,
    state_setter=DEFAULT_STATE_SETTER,
)
if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    new_model = PPO(
        **DEFAULT_PPO_HYPERPARAMS,
        env=env,
    )
    new_model.set_parameters(
        model.get_parameters()  # pyright: ignore[reportArgumentType]
    )
    model = train_model(
        new_model,
        model_name=model_name,
        total_timesteps=4_000_000,
        progress_bar=False,
    )

# %% [markdown]
# ### Train PPO 3v3
# Trains an agent in a 3v3 self-play scenario.

# %%
# @title Train PPO 3v3 Self-Play
# @markdown Trains an agent in a 3v3 self-play scenario.
model_name = "ppo-3v3_Self-Play_v1"
env = make_env(
    team_size=3,
    bots=None,
    state_setter=DEFAULT_STATE_SETTER,
)
if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    new_model = PPO(
        **DEFAULT_PPO_HYPERPARAMS,
        env=env,
    )
    new_model.set_parameters(
        model.get_parameters()  # pyright: ignore[reportArgumentType]
    )
    model = train_model(
        new_model,
        model_name=model_name,
        total_timesteps=6_000_000,
    )

# %%
# @title Gym Debug
# @markdown Debugging utility to step through the environment and verify logic.
# https://wazarr94.github.io/
game = Game(folder_rec=FOLDER_REC, enable_renderer=False)
env = haxballgym.make(
    game=game,
    tick_skip=TICK_SKIP,
    team_size=1,
    bots=[],
    # bots=[BOTS["ChaseBot"](1)],
    reward_fn=DEFAULT_REWARD_FN,
    obs_builder=DEFAULT_OBS_BUILDER,
    state_setter=RandomKickoff(red_percent=1.0),
)

model_name = "ppo-1v0_StrikerDrill_v1"
model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip")

ep_reward = 0
obs, info = env.reset()
for step in range(20):
    # for step in range(20_000):
    # print(step, obs)
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    ep_reward += rewards
    print(step, action, rewards)

    # actions = [model.predict(o)[0] for o in obs]
    # obs, rewards, terminated, truncated, info = env.step(actions)
    # print(step, actions, rewards)
    if terminated or truncated:
        break
obs, info = env.reset(options={"save_recording": True})
print(ep_reward)

# %% [markdown]
# ## League Training

# %%
# @title Define League Constants
LEAGUE_WORKER = "LEAGUE-WORKER"
LEAGUE_START_MODEL = "ppo-1v0_StrikerDrill_v0"  # @param {"type":"string","placeholder":"ppo-1v0_StrikerDrill_v0"}


# %%
# @title Define Sync League Models
def sync_league_models():
    """Fetches the latest models from Drive to ensure we fight the newest meta."""
    print("Syncing latest league opponents from Google Drive...")

    # Re-query Drive for the current state of the folder
    file_list = drive.ListFile(
        {
            "q": f"'{FOLDER_MODELS_ID}' in parents and mimeType='application/zip' and trashed=false"
        }
    ).GetList()

    synced_bots: list[type[Bot]] = []
    for model_file in file_list:
        file_name = model_file["title"]
        MODELS[file_name] = model_file
        if file_name.startswith(LEAGUE_WORKER):
            bot = load_model(model_file)
            synced_bots.append(bot)

    return synced_bots


# %%
# @title Define LeagueOpponentBot
class LeagueBot(Bot):
    """
    A bot for League Training.
    Swaps to a random model every time the environment resets.
    """

    def __init__(self, bot_classes: list[type[Bot]], tick_skip: int = TICK_SKIP):
        super().__init__(tick_skip=tick_skip)
        self._rng = np.random.default_rng()
        self.bot_classes = bot_classes
        self.current_bot = bot_classes[0](tick_skip=tick_skip)

    def step_game(self, player: PlayerHandler, game: Game):
        if game.score.time == 0:
            self.current_bot = self.bot_classes[
                self._rng.choice(len(self.bot_classes))
            ](tick_skip=self.tick_skip)
        return self.current_bot.step_game(player, game)


# %%
# @title Define League Training
def train_league(
    model: BaseAlgorithm,
    model_name: str,
    static_bots: list[type[Bot]],
    rounds: int,
    timesteps_per_round: int,
    team_size: int = 3,
    **env_kwargs,
):
    for r in range(rounds):
        print(f"\n--- LEAGUE ROUND {r + 1}/{rounds} ---")

        # Fetch latest opponent models from Drive
        dynamic_league_bots = sync_league_models()

        # Bind env
        bots = [
            LeagueBot(static_bots + dynamic_league_bots, tick_skip=TICK_SKIP)
            for _ in range(team_size)
        ]
        new_env = make_env(
            team_size=team_size,
            bots=bots,  # pyright: ignore[reportArgumentType]
            reward_fn=DEFAULT_REWARD_FN,
            **env_kwargs,
        )
        new_env.reset()

        # Train
        model.set_env(new_env)
        model.learn(
            total_timesteps=timesteps_per_round,
            tb_log_name=model_name,
            reset_num_timesteps=False,
        )

        # Upload new model to Drive
        save_model(model, model_name)
        save_tb_logs(model)

    return model


# %%
# @title Start League Training

# 1. Define what to start from
MY_WORKER_NAME = "G1"  # @param {"type":"string","placeholder":"League-Worker-A"}
model_name = f"{LEAGUE_WORKER}-{MY_WORKER_NAME}"
env = make_env(
    team_size=1,
    bots=[],
    state_setter=DEFAULT_STATE_SETTER,
)

if model_name in BOTS:
    model = PPO.load(f"{FOLDER_MODELS}/{model_name}.zip", env)
else:
    model = PPO.load(f"{FOLDER_MODELS}/{LEAGUE_START_MODEL}.zip", env)

# 2. Define the static opponents
STATIC_LEAGUE_BOTS = [
    BOTS["RandomBot"],
    BOTS["RandomBot"],
    BOTS["RandomBot"],
    BOTS["RandomBot"],
    BOTS["RandomBot"],
    BOTS["RandomBot"],
    BOTS["GoalkeeperBot"],
    BOTS["GoalkeeperBot"],
    BOTS["StrikerBot"],
    BOTS["StrikerBot"],
]

# Start League Training
rounds = 50  # @param {"type":"integer","placeholder":"50"}
timesteps_per_round = 200_000  # @param {"type":"integer","placeholder":"200_000"}
train_league(
    model=model,
    model_name=model_name,
    team_size=1,
    static_bots=STATIC_LEAGUE_BOTS,
    rounds=rounds,
    timesteps_per_round=timesteps_per_round,
)

# %% [markdown]
# # Evaluation
#
# Runs matches, records gameplay, and computes comparative ratings across team configurations.

# %%
# @title Define Teams
# @markdown Builds team compositions from available bots for head-to-head matches and tournaments.
Team = dict[str, int]
Teams = dict[str, Team]

TEAMS_1: Teams = dict([(f"{bot}-1", {bot: 1}) for bot in BOTS])
TEAMS_2: Teams = dict([(f"{bot}-2", {bot: 2}) for bot in BOTS])
TEAMS_3: Teams = dict([(f"{bot}-3", {bot: 3}) for bot in BOTS])
TEAMS_3.update(
    [(f"GoalkeeperBot-1_StrikerBot-2", {"GoalkeeperBot": 1, "StrikerBot": 2})]
)

TEAMS: Teams = {**TEAMS_1, **TEAMS_2, **TEAMS_3}
TEAMS["Empty-0"] = {}

print("\n".join(TEAMS))


# %%
# @title Run Game
# @markdown Simulates a single match between two teams and returns the score. View game recordings at <https://wazarr94.github.io/>
def run_game(
    red_team: Team | str,
    blue_team: Team | str,
    *,
    step_limit: int = 2_000,
    score_limit: int = 3,
    enable_renderer: bool = False,
    enable_recorder: bool = False,
    folder_rec: str = "",
):
    if not folder_rec and isinstance(red_team, str) and isinstance(blue_team, str):
        folder_rec = f"{red_team}_vs_{blue_team}"
    folder_rec = f"{FOLDER_REC}/{folder_rec}"

    if isinstance(red_team, str):
        red_team = TEAMS[red_team]
    if isinstance(blue_team, str):
        blue_team = TEAMS[blue_team]

    game = Game(
        folder_rec=folder_rec,
        enable_renderer=enable_renderer,
        fov=420,
        enable_recorder=enable_recorder,
    )
    game.score = GameScore(score_limit=score_limit)

    for bot, count in red_team.items():
        for i in range(count):
            game.add_player(
                PlayerHandler(f"{bot}_{i}", TeamID.RED, BOTS[bot](tick_skip=TICK_SKIP))
            )
    for bot, count in blue_team.items():
        for i in range(count):
            game.add_player(
                PlayerHandler(f"{bot}_{i}", TeamID.BLUE, BOTS[bot](tick_skip=TICK_SKIP))
            )

    game.start()
    for _ in range(step_limit):
        if game.step([player.step(game) for player in game.players]):
            break

    score = copy.deepcopy(game.score)
    game.stop(save_recording=enable_recorder)
    return score


red_team = (
    "StrikerBot-1"  # @param {"type":"string","placeholder":"ppo-1v0_StrikerDrill_v0-1"}
)
blue_team = (
    "ppo-1v0_StrikerDrill_v0-1"  # @param {"type":"string","placeholder":"Empty-0"}
)
score = run_game(
    red_team,
    blue_team,
    step_limit=6_000,
    # score_limit = 2,
    score_limit=5,
    enable_recorder=True,
)
print(score.get_score_string())

# %%
# @title Record Game
# @markdown Records a match to video file for visual analysis.
import subprocess

from IPython.display import Video
from pyvirtualdisplay import Display  # pyright: ignore[reportPrivateImportUsage]


def record_game(
    red_team: Team | str,
    blue_team: Team | str,
    filename: str = "",
    size: tuple[int, int] = (420, 200),
    **config,
):
    if not filename and isinstance(red_team, str) and isinstance(blue_team, str):
        filename = f"{red_team}_vs_{blue_team}"
    filename = f"{FOLDER_REC}/{filename}.mp4"

    display = Display(visible=False, size=size)
    display.start()

    ffmpeg_process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "x11grab",
            "-draw_mouse",
            "0",
            "-r",
            "30",
            "-s",
            f"{size[0]}x{size[1]}",
            "-i",
            display.new_display_var,
            # "-preset",
            # "ultrafast",
            filename,
        ]
    )

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=lambda: queue.put(
            run_game(
                red_team,
                blue_team,
                enable_renderer=True,
                **config,
            )
        )
    )
    process.start()
    process.join()

    ffmpeg_process.terminate()
    try:
        ffmpeg_process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        ffmpeg_process.kill()
    display.stop()
    if process.is_alive():
        process.terminate()

    return queue.get(), Video(filename, embed=True)


# red_team = {'GoalkeeperBot': 1, 'ChaseBot': 1}
red_team = "ppo-1v0_StrikerDrill_v0-1"
blue_team = "Empty-0"
score, video = record_game(
    red_team,
    blue_team,
    filename="Game1",
    size=(420, 200),
    # size=(840, 400),
    step_limit=2_000,
)
video

# %%
# @title Run Tournament
# @markdown Conducts a tournament between defined teams and calculates Elo/TrueSkill ratings.
from concurrent.futures import ProcessPoolExecutor

from openskill.models import PlackettLuce


def run_tournament(teams: Teams = TEAMS, **config):
    model = PlackettLuce()
    ratings = {name: model.rating(name=name) for name in teams}

    matchups = [(red, blue) for red in teams for blue in teams if red != blue]
    print(f"Running {len(matchups)} matchups...")

    with ProcessPoolExecutor() as executor:
        future_to_match = {
            executor.submit(
                run_game,
                teams[red],
                teams[blue],
                **config,
            ): (red, blue)
            for red, blue in matchups
        }

        for future in future_to_match:
            red, blue = future_to_match[future]
            score = future.result()
            print(f"{red} vs {blue}: {score.red} - {score.blue}")

            [ratings[red]], [ratings[blue]] = model.rate(
                [[ratings[red]], [ratings[blue]]], scores=[score.red, score.blue]
            )

    return ratings


ratings = run_tournament(
    TEAMS_2,
    step_limit=10_000,
    score_limit=5,
)

for name, rating in sorted(ratings.items(), key=lambda x: x[1].ordinal(), reverse=True):
    print(f"{name}: {rating.ordinal()}")
