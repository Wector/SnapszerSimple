# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A parity test for the JAX and Python implementations of Snapszer."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import snapszer_base
import snapszer_jax


class ParityTest(parameterized.TestCase):

  @parameterized.parameters(
      {"seed": 1, "num_actions": 10},
      {"seed": 42, "num_actions": 50},
  )
  def test_parity(self, seed: int, num_actions: int):
    """Tests that the JAX and Python implementations are in sync."""
    # Print the available JAX devices.
    print(f"JAX devices: {jax.devices()}")

    # Create the initial states.
    base_state = snapszer_base.SnapszerState.new(seed)
    jax_game = snapszer_jax.Snapszer()
    jax_state = snapszer_jax.State.new_from_deck(
        jnp.array(base_state.deck)
    )

    # Check that the initial states are in sync.
    self.assert_states_equal(base_state, jax_state)

    # Play a few random actions and check that the states are still in sync.
    for _ in range(num_actions):
      if base_state.is_terminal():
        break

      # Get the legal actions from both implementations.
      base_legal_actions = base_state.legal_actions()
      jax_legal_actions = np.where(
          jax_game.legal_actions(jax_state)
      )[0].tolist()

      # Check that the legal actions are the same.
      self.assertEqual(set(base_legal_actions), set(jax_legal_actions))

      # Choose a random action to play.
      action = np.random.choice(base_legal_actions)

      # Apply the action to both states.
      base_state.apply_action(action)
      jax_state = jax_game.step(jax_state, action)

      # Check that the states are still in sync.
      self.assert_states_equal(base_state, jax_state)

  def assert_states_equal(
      self,
      base_state: snapszer_base.SnapszerState,
      jax_state: snapszer_jax.State,
  ):
    """Asserts that the two states are equal."""
    self.assertEqual(base_state.trump, jax_state.trump)
    self.assertEqual(base_state.trump_card, jax_state.trump_card)
    np.testing.assert_array_equal(
        np.array(base_state.deck), jax_state.deck
    )
    np.testing.assert_array_equal(
        np.array(base_state.stock), jax_state.stock
    )
    self.assertEqual(base_state.stock_idx, jax_state.stock_idx)
    for i in range(2):
      np.testing.assert_array_equal(
        np.array(base_state.hands[i]),
        jax_state.hands[i][jax_state.hands[i] != -1],
      )
    np.testing.assert_array_equal(
        np.array(base_state.hand_masks), jax_state.hand_masks
    )
    self.assertEqual(base_state.current_player, jax_state.current_player)
    self.assertEqual(base_state.leader, jax_state.leader)
    np.testing.assert_array_equal(
        np.array(
            [-1 if c is None else c for c in base_state.trick_cards]
        ),
        jax_state.trick_cards,
    )
    np.testing.assert_array_equal(
        np.array(base_state.points), jax_state.points
    )
    np.testing.assert_array_equal(
        np.array(base_state.tricks_won), jax_state.tricks_won
    )
    self.assertEqual(base_state.closed, jax_state.closed)
    self.assertEqual(
        -1 if base_state.closed_by is None else base_state.closed_by,
        jax_state.closed_by,
    )
    self.assertEqual(base_state.trump_taken, jax_state.trump_taken)
    self.assertEqual(
        -1
        if base_state.last_trick_winner is None
        else base_state.last_trick_winner,
        jax_state.last_trick_winner,
    )
    for i in range(2):
      for j in range(snapszer_jax.NUM_SUITS):
        self.assertEqual(
            base_state.marriages_scored[i][j],
            jax_state.marriages_scored[i, j],
        )
    self.assertEqual(base_state.terminal, jax_state.terminal)
    self.assertEqual(
        -1 if base_state.winner is None else base_state.winner, jax_state.winner
    )
    if base_state.game_points is None:
      np.testing.assert_array_equal(
          np.zeros(2, dtype=np.int32), jax_state.game_points
      )
    else:
      np.testing.assert_array_equal(
          np.array(base_state.game_points), jax_state.game_points
      )


if __name__ == "__main__":
  absltest.main()
