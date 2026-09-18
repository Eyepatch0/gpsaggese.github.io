# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AlphaZero Game Representation API
#
# - Reuse the shared Tic-Tac-Toe rules
# - Inspect the three functions in `alphazero_utils.py` line by line
# - Explore the inputs and outcomes used by search and learning
# - See [README.md](README.md) for setup and API conventions

# %% [markdown]
# ## Imports and Setup
#
# - Launch from the repository environment documented in [README.md](README.md)
# - The repository and `helpers_root` must be on `PYTHONPATH`

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np

import helpers.hdbg as hdbg
import research.Implement_AlphaZero.alphazero_utils as rialzut
import research.Implement_MonteCarlo_Tree_Search_and_Alpha_Zero.game_examples as rimtsaazge

hdbg.init_logger(verbosity=logging.INFO)
_LOG = logging.getLogger(__name__)

# %% [markdown]
# # Part 1: States and Actions
#
# | API | Result | Purpose |
# | :--- | :--- | :--- |
# | `encode_state(game, state)` | Flat `float32` vector | Model input from the current player's perspective |
# | `get_legal_action_mask(game, state, action_size)` | Boolean vector | Legal entries in a fixed policy output |
# | `get_terminal_value(game, state)` | Float or `None` | Exact terminal outcome, distinct from unfinished play |

# %% [markdown]
# ## Cell 1.1: Reuse the Game
#
# - The original state uses `1` for X, `-1` for O, and `0` for empty
# - Actions are row-major indices: top row `0, 1, 2`, middle `3, 4, 5`,
#   bottom `6, 7, 8`

# %%
# Print is intentional throughout: this notebook exposes the API results.
game = rimtsaazge.TicTacToe()
state = game.get_initial_state()
print("state=", state)
print("board=\n" + game.render(state))
print("legal_moves=", game.get_legal_moves(state))

# %% [markdown]
# ## Cell 1.2: Encode the Empty Board
#
# - The model input is a flat vector of nine numbers
# - An encoding is a fresh array; the tuple state remains owned by the game

# %%
# Inspect the model input shape and dtype.
encoded = rialzut.encode_state(game, state)
print("encoded=", encoded)
print("shape=", encoded.shape, "dtype=", encoded.dtype)
np.testing.assert_array_equal(encoded, [0] * 9)

# %% [markdown]
# ## Cell 1.3: Change the Player's Perspective
#
# - X plays the center, then O is the player to move
# - O sees X's center piece as `-1`; its own pieces are encoded as `+1`
# - Multiply cell signs by the current player; keep cell positions unchanged
# - Try changing `move` from `4` to another index and rerun this cell

# %%
# Always start this experiment from an empty board so it can be rerun.
move = 4
state = game.apply_move(game.get_initial_state(), move)
encoded = rialzut.encode_state(game, state)
print("current_player=", game.get_current_player(state))
print("board=\n" + game.render(state))
print("encoded=", encoded)
hdbg.dassert_eq(encoded[move], -1.0, "O sees X's piece as an opponent")

# %% [markdown]
# - Never feed `encoded` into `apply_move()` or other game-rule methods
# - Rules use the original `state`; encoding supplies the model input

# %% [markdown]
# ## Cell 1.4: Preserve a Fixed Action Space
#
# - All nine policy entries retain their original indices
# - The occupied cell is False, but the mask still has length nine
# - A mask indicates legality; a policy distribution assigns probabilities

# %%
# Compare the fixed-size mask with the game's shorter list of legal moves.
mask = rialzut.get_legal_action_mask(game, state, 9)
print("legal_action_mask=", mask)
print("legal_indices=", np.flatnonzero(mask))
np.testing.assert_array_equal(np.flatnonzero(mask), game.get_legal_moves(state))

# %% [markdown]
# # Part 2: Exact Outcomes

# %% [markdown]
# ## Cell 2.1: An Unfinished Game Has No Exact Value Yet
#
# - `None` means unfinished; it does not mean a draw
# - Nonterminal positions require value estimates rather than exact outcomes

# %%
# The one-move board has no terminal outcome.
value = rialzut.get_terminal_value(game, state)
print("terminal_value=", value)
hdbg.dassert_is(value, None, "An unfinished game has no exact outcome")

# %% [markdown]
# ## Cell 2.2: Walk Through a Complete Game
#
# - Use the specified moves `0, 3, 1, 4, 2`; X wins across the top row
# - Check every action against the mask before applying it
# - These are hand-chosen moves, not an agent or self-play data collector

# %%
# Trace the public game and representation APIs.
state = game.get_initial_state()
for move in [0, 3, 1, 4, 2]:
    mask = rialzut.get_legal_action_mask(game, state, 9)
    hdbg.dassert(mask[move], "The demonstration must use a legal move")
    state = game.apply_move(state, move)
    print(
        "move=",
        move,
        "terminal_value=",
        rialzut.get_terminal_value(game, state),
    )
print("final_board=\n" + game.render(state))

# %% [markdown]
# ## Cell 2.3: Interpret a Win From the Next Player's Perspective
#
# - Winner: X (`1`); next player under the game convention: O (`-1`)
# - Value: winner times next player, hence `-1.0`
# - Negate this value to express the outcome from X's perspective
# - The game is over: empty cells must also be masked
# - The existing MCTS stores values for the player who entered a node;
#   its convention differs from this player-to-move value

# %%
# Confirm the exact outcome and legal-action mask at a terminal state.
value = rialzut.get_terminal_value(game, state)
mask = rialzut.get_legal_action_mask(game, state, 9)
print("winner=", game.get_winner(state))
print("next_player=", game.get_current_player(state))
print("terminal_value=", value)
print("legal_action_mask=", mask)
hdbg.dassert_eq(value, -1.0, "O has lost after X wins")
np.testing.assert_array_equal(mask, [False] * 9)

# %% [markdown]
# ## Cell 2.4: Distinguish a Draw
#
# - This full board has no winning line
# - A completed draw returns `0.0`, unlike the unfinished game's `None`

# %%
# Inspect a reachable drawn position.
draw_state = (1, -1, 1, 1, -1, -1, -1, 1, 1)
value = rialzut.get_terminal_value(game, draw_state)
print("draw_board=\n" + game.render(draw_state))
print("terminal_value=", value)
hdbg.dassert_eq(value, 0.0, "A draw has zero value")
