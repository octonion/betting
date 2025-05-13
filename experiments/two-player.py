import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class PooledKellyBettingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, initial_bankroll_A=250.0, initial_bankroll_B=250.0,
                 initial_flips_A=20, initial_flips_B=20, prob_heads=0.6):
        super(PooledKellyBettingEnv, self).__init__()

        self.initial_bankroll_A = float(initial_bankroll_A)
        self.initial_bankroll_B = float(initial_bankroll_B)
        self.initial_flips_A = initial_flips_A
        self.initial_flips_B = initial_flips_B
        self.prob_heads = prob_heads
        self.kelly_fraction = (prob_heads * 1 - (1 - prob_heads) * 1) / 1 # b=1 for 1:1 odds

        # Current state
        self.bankroll_A = self.initial_bankroll_A
        self.bankroll_B = self.initial_bankroll_B
        self.flips_A = self.initial_flips_A
        self.flips_B = self.initial_flips_B
        self.current_total_flips_made = 0
        self.max_total_flips = self.initial_flips_A + self.initial_flips_B

        # Define action space: Discrete(2) -> 0: Player A bets, 1: Player B bets
        self.action_space = spaces.Discrete(2)

        # Define observation space: [bankroll_A, bankroll_B, flips_A, flips_B]
        # Normalization is crucial here for real applications.
        # Using high values for observation bounds as a placeholder.
        # A better approach would be dynamic normalization or log transforms.
        low_obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_obs = np.array([1e6, 1e6, float(self.initial_flips_A), float(self.initial_flips_B)], dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        # For rendering (optional)
        self.render_mode = None


    def _get_obs(self):
        # Basic normalization (dividing by initial total for bankrolls)
        # This is very simplistic; more advanced normalization would be needed.
        # obs_ba = self.bankroll_A / (self.initial_bankroll_A + self.initial_bankroll_B)
        # obs_bb = self.bankroll_B / (self.initial_bankroll_A + self.initial_bankroll_B)
        # Using raw values for now, but scaled down for initial high_obs
        return np.array([self.bankroll_A, self.bankroll_B,
                         float(self.flips_A), float(self.flips_B)], dtype=np.float32)

    def _get_info(self):
        return {
            "bankroll_A": self.bankroll_A, "bankroll_B": self.bankroll_B,
            "flips_A": self.flips_A, "flips_B": self.flips_B,
            "total_flips_made": self.current_total_flips_made
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility

        self.bankroll_A = self.initial_bankroll_A
        self.bankroll_B = self.initial_bankroll_B
        self.flips_A = self.initial_flips_A
        self.flips_B = self.initial_flips_B
        self.current_total_flips_made = 0

        if self.render_mode == "human":
            print(f"Env Reset: BA={self.bankroll_A:.2f}, BB={self.bankroll_B:.2f}, NA={self.flips_A}, NB={self.flips_B}")

        return self._get_obs(), self._get_info()

    def step(self, action):
        terminated = False
        truncated = False # For time limit, not used here but good practice

        # Determine which player is chosen by the action
        chosen_player_idx = action # 0 for A, 1 for B

        player_to_bet = -1 # 0 for A, 1 for B
        can_A_bet = self.flips_A > 0 and self.bankroll_A > 1e-6 # Min bettable amount
        can_B_bet = self.flips_B > 0 and self.bankroll_B > 1e-6

        if chosen_player_idx == 0: # Agent chose A
            if can_A_bet:
                player_to_bet = 0
            elif can_B_bet: # A cannot bet, B takes over
                player_to_bet = 1
        elif chosen_player_idx == 1: # Agent chose B
            if can_B_bet:
                player_to_bet = 1
            elif can_A_bet: # B cannot bet, A takes over
                player_to_bet = 0
        
        # If only one player can bet overall, that player bets
        if not can_A_bet and can_B_bet:
            player_to_bet = 1
        elif not can_B_bet and can_A_bet:
            player_to_bet = 0
        elif not can_A_bet and not can_B_bet: # No one can bet
            terminated = True
            # Reward is calculated at the end
            # If bankrolls are zero, log can be problematic. Add small epsilon.
            final_pooled_wealth = self.bankroll_A + self.bankroll_B
            reward = math.log(final_pooled_wealth + 1e-9) # Avoid log(0)
            if self.render_mode == "human":
                 print(f"End of Episode (no one can bet). Final W_total={final_pooled_wealth:.2f}. Reward={reward:.2f}")
            return self._get_obs(), reward, terminated, truncated, self._get_info()


        # Calculate bet amount based on Kelly for pooled wealth
        total_wealth = self.bankroll_A + self.bankroll_B
        if total_wealth < 1e-6: # Effectively bankrupt
            terminated = True
            reward = math.log(1e-9) # Very low reward
            if self.render_mode == "human":
                 print(f"End of Episode (total wealth near zero). Reward={reward:.2f}")
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        target_bet_amount = self.kelly_fraction * total_wealth
        
        actual_bet_amount = 0
        bettor_name = ""

        if player_to_bet == 0: # Player A bets
            bettor_name = "A"
            actual_bet_amount = min(target_bet_amount, self.bankroll_A)
            if actual_bet_amount < 1e-6 : # effectively cannot make a meaningful bet
                actual_bet_amount = 0 # consider it no bet
            self.flips_A -= 1
        elif player_to_bet == 1: # Player B bets
            bettor_name = "B"
            actual_bet_amount = min(target_bet_amount, self.bankroll_B)
            if actual_bet_amount < 1e-6 : # effectively cannot make a meaningful bet
                actual_bet_amount = 0 # consider it no bet
            self.flips_B -= 1
        else: # Should not happen if logic above is correct and at least one can bet
            # This case means no valid player was chosen to bet, though game not necessarily over
            # This might happen if chosen player cannot make a meaningful bet (e.g. bankroll too low)
             pass # No bet this turn, effectively state doesn't change much


        # Simulate coin flip
        outcome_win = False
        if actual_bet_amount > 0: # Only if a bet was made
            if self.np_random.random() < self.prob_heads:
                outcome_win = True

            # Update bankroll of the bettor
            if player_to_bet == 0:
                self.bankroll_A += actual_bet_amount if outcome_win else -actual_bet_amount
                self.bankroll_A = max(0.0, self.bankroll_A) # Cannot go below zero
            elif player_to_bet == 1:
                self.bankroll_B += actual_bet_amount if outcome_win else -actual_bet_amount
                self.bankroll_B = max(0.0, self.bankroll_B)
        
        self.current_total_flips_made += 1
        reward = 0.0 # Intermediate reward is 0

        if self.render_mode == "human":
            print(f"Flip {self.current_total_flips_made}: Player {bettor_name} chose to bet.")
            print(f"  Target: {target_bet_amount:.2f}, Actual Bet: {actual_bet_amount:.2f}")
            if actual_bet_amount > 0:
                print(f"  Outcome: {'Win' if outcome_win else 'Loss'}")
            print(f"  State: BA={self.bankroll_A:.2f}, BB={self.bankroll_B:.2f}, NA={self.flips_A}, NB={self.flips_B}")


        # Check for termination conditions
        if self.flips_A == 0 and self.flips_B == 0:
            terminated = True
        if self.current_total_flips_made >= self.max_total_flips : # Redundant if flips are main limit
             terminated = True

        if terminated:
            final_pooled_wealth = self.bankroll_A + self.bankroll_B
            reward = math.log(final_pooled_wealth + 1e-9) # Avoid log(0)
            if self.render_mode == "human":
                 print(f"End of Episode. Final W_total={final_pooled_wealth:.2f}. Reward={reward:.2f}")


        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self, mode='human'):
        self.render_mode = mode # Store for step function
        # No complex rendering, will print in step if mode is human

    def close(self):
        pass

if __name__ == '__main__':
    # Instantiate and check the environment (optional but good practice)
    env = PooledKellyBettingEnv(initial_flips_A=5, initial_flips_B=5) # Short episodes
    # check_env(env) # This might complain about observation normalization if not perfect

    # To run a few random steps (for testing the environment)
    print("--- Testing Environment with Random Actions ---")
    obs, info = env.reset()
    env.render_mode = "human" # Enable printouts
    for i in range(12): # A few more than total flips
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps. Final Reward: {reward}")
            obs, info = env.reset()
            # break # uncomment to stop after one episode

    env.render_mode = None # Disable printouts for training

    print("\n--- Training PPO Agent ---")
    # Re-initialize environment for training
    train_env = PooledKellyBettingEnv(initial_flips_A=20, initial_flips_B=20)

    # Parameters for PPO are defaults here, would need tuning for real performance
    # For MLP (Multi-Layer Perceptron) policy
    # Set verbose=1 to see training logs
    model = PPO("MlpPolicy", train_env, verbose=0, n_steps=200, batch_size=50, n_epochs=5, learning_rate=3e-4, device="cpu")

    # Training - total_timesteps is very small to test
    # For a real task, this would be much larger (e.g., 1e6, 1e7 or more)
    # and might take hours/days.
    try:
        model.learn(total_timesteps=20000) # Increased for a tiny bit more training
        print("Training finished (or interrupted).")

        # Save the model (optional)
        # model.save("ppo_pooled_kelly")
        # print("Model saved.")

        # Load the model (optional)
        # model = PPO.load("ppo_pooled_kelly")
        # print("Model loaded.")

        # --- Evaluate the trained agent ---
        print("\n--- Evaluating Trained Agent ---")
        eval_env = PooledKellyBettingEnv(initial_flips_A=20, initial_flips_B=20)
        obs, info = eval_env.reset()
        eval_env.render_mode = "human"
        cumulative_reward = 0
        n_eval_episodes = 5

        for episode in range(n_eval_episodes):
            terminated = False
            truncated = False
            ep_reward = 0
            print(f"\n--- Evaluation Episode {episode+1} ---")
            obs, info = eval_env.reset() # Reset before each episode
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward # In our case, only terminal reward matters
                if terminated or truncated:
                    print(f"Episode finished. Final Reward: {ep_reward}")
                    cumulative_reward += ep_reward # ep_reward is the terminal log utility
                    break
        print(f"\nAverage reward over {n_eval_episodes} evaluation episodes: {cumulative_reward / n_eval_episodes:.2f}")

    except Exception as e:
        print(f"An error occurred during training or evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("--- Script Finished ---")
