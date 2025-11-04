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

"""A JAX implementation of the Hungarian Snapszer card game."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from functools import partial


# Constants for the game
SUITS = jnp.array([0, 1, 2, 3])
SUIT_NAMES = {0: "S", 1: "H", 2: "D", 3: "C"}
RANKS = jnp.array([0, 1, 2, 3, 4])  # A, 10, K, Q, J
RANK_NAMES = {0: "A", 1: "10", 2: "K", 3: "Q", 4: "J"}
RANK_STRENGTH = jnp.array([5, 4, 3, 2, 1])
CARD_POINTS = jnp.array([11, 10, 4, 3, 2])
NUM_SUITS = len(SUITS)
NUM_RANKS = len(RANKS)
NUM_CARDS = NUM_SUITS * NUM_RANKS
TRICKS_PER_GAME = NUM_CARDS // 2
TRUMP_JACK_RANK = NUM_RANKS - 1
EXCHANGE_TRUMP_ACTION = NUM_CARDS
CLOSE_TALON_ACTION = NUM_CARDS + 1
TOTAL_ACTIONS = NUM_CARDS + 2
CLOSE_MIN_ABOVE_TRUMP = 3
OBSERVATION_SIZE = 80


# Helper functions for card operations
def card_id(suit: int, rank: int) -> int:
  """Returns the unique ID of a card."""
  return suit * NUM_RANKS + rank


def cid_suit(cid: int) -> int:
  """Returns the suit of a card from its ID."""
  return cid // NUM_RANKS


def cid_rank(cid: int) -> int:
  """Returns the rank of a card from its ID."""
  return cid % NUM_RANKS


def card_points(cid: int) -> int:
  """Returns the point value of a card."""
  return CARD_POINTS[cid_rank(cid)]


def card_str(cid: int) -> str:
  """Returns the string representation of a card."""
  return f"{RANK_NAMES[cid_rank(cid)]}{SUIT_NAMES[cid_suit(cid)]}"


def cards_to_mask(cards: jnp.ndarray) -> int:
  """Converts a list of cards to a bitmask."""
  mask = 0
  for cid in cards:
    mask |= 1 << cid
  return mask


def mask_contains(mask: int, cid: int) -> bool:
  """Checks if a card is in a bitmask."""
  return (mask >> cid) & 1 == 1


class State(NamedTuple):
  """A JAX-compatible state for the Snapszer game."""

  trump: jnp.ndarray
  trump_card: jnp.ndarray
  deck: jnp.ndarray
  stock: jnp.ndarray
  stock_idx: jnp.ndarray
  hands: jnp.ndarray
  hand_masks: jnp.ndarray
  current_player: jnp.ndarray
  leader: jnp.ndarray
  trick_cards: jnp.ndarray
  points: jnp.ndarray
  tricks_won: jnp.ndarray
  closed: jnp.ndarray
  closed_by: jnp.ndarray
  trump_taken: jnp.ndarray
  last_trick_winner: jnp.ndarray
  marriages_scored: jnp.ndarray
  terminal: jnp.ndarray
  winner: jnp.ndarray
  game_points: jnp.ndarray

  @staticmethod
  def new(key: jax.random.PRNGKey) -> State:
    """Creates a new game state."""
    deck = jnp.arange(NUM_CARDS)
    key, subkey = jax.random.split(key)
    deck = jax.random.permutation(subkey, deck)
    return State.new_from_deck(deck)

  @staticmethod
  def new_from_deck(deck: jnp.ndarray) -> State:
    """Creates a new game state from a deck."""
    p0 = jnp.sort(deck[:5])
    p1 = jnp.sort(deck[5:10])
    hands = jnp.array([p0, p1])
    hands = jnp.pad(hands, ((0, 0), (0, 0)), constant_values=-1)
    trump_card = deck[10]
    trump = cid_suit(trump_card)
    stock = deck[11:]

    return State(
        trump=jnp.array(trump, dtype=jnp.int32),
        trump_card=jnp.array(trump_card, dtype=jnp.int32),
        deck=deck,
        stock=stock,
        stock_idx=jnp.array(0, dtype=jnp.int32),
        hands=hands,
        hand_masks=jnp.array(
            [cards_to_mask(p0), cards_to_mask(p1)], dtype=jnp.int32
        ),
        current_player=jnp.array(0, dtype=jnp.int32),
        leader=jnp.array(0, dtype=jnp.int32),
        trick_cards=jnp.full(2, -1, dtype=jnp.int32),
        points=jnp.zeros(2, dtype=jnp.int32),
        tricks_won=jnp.zeros(2, dtype=jnp.int32),
        closed=jnp.array(False, dtype=jnp.bool_),
        closed_by=jnp.array(-1, dtype=jnp.int32),
        trump_taken=jnp.array(False, dtype=jnp.bool_),
        last_trick_winner=jnp.array(-1, dtype=jnp.int32),
        marriages_scored=jnp.zeros((2, NUM_SUITS), dtype=jnp.bool_),
        terminal=jnp.array(False, dtype=jnp.bool_),
        winner=jnp.array(-1, dtype=jnp.int32),
        game_points=jnp.zeros(2, dtype=jnp.int32),
    )


def _stock_remaining(state: State) -> jnp.ndarray:
  """Returns the number of cards remaining in the stock."""
  return len(state.stock) - state.stock_idx


def _talon_cards_remaining(state: State) -> jnp.ndarray:
  """Returns the number of cards remaining in the talon."""
  return _stock_remaining(state) + (1 - state.trump_taken)


def _talon_empty(state: State) -> jnp.ndarray:
  """Returns true if the talon is empty."""
  return state.trump_taken & (state.stock_idx >= len(state.stock))


def _strict_rules_active(state: State) -> jnp.ndarray:
  """Returns true if strict rules are active."""
  return state.closed | _talon_empty(state)


def _trick_winner(
    lead_cid: jnp.ndarray, reply_cid: jnp.ndarray, trump: jnp.ndarray
) -> jnp.ndarray:
  """Determines the winner of a trick."""
  ls, lr = cid_suit(lead_cid), cid_rank(lead_cid)
  rs, rr = cid_suit(reply_cid), cid_rank(reply_cid)

  is_same_suit = rs == ls
  is_reply_trump = rs == trump
  is_lead_trump = ls == trump

  winner_if_same_suit = jnp.where(
      RANK_STRENGTH[lr] >= RANK_STRENGTH[rr], 0, 1
  )
  winner_if_different_suit = jnp.where(is_reply_trump & ~is_lead_trump, 1, 0)

  return jnp.where(
      is_same_suit, winner_if_same_suit, winner_if_different_suit
  )


def _legal_reply_mask(
    hand_mask: jnp.ndarray, lead_cid: jnp.ndarray, trump: jnp.ndarray
) -> jnp.ndarray:
  """Returns a mask of legal cards to play in response to a lead."""
  lead_s = cid_suit(lead_cid)
  lead_r = cid_rank(lead_cid)
  lead_strength = RANK_STRENGTH[lead_r]

  cids = jnp.arange(NUM_CARDS)
  suits = cids // NUM_RANKS
  ranks = cids % NUM_RANKS
  strengths = RANK_STRENGTH[ranks]

  in_hand = (hand_mask & (1 << cids)) > 0
  same_suit_in_hand = in_hand & (suits == lead_s)
  beating_in_hand = same_suit_in_hand & (strengths > lead_strength)
  trumps_in_hand = in_hand & (suits == trump)

  same_suit_mask = jnp.sum(jnp.where(same_suit_in_hand, 1 << cids, 0))
  beating_mask = jnp.sum(jnp.where(beating_in_hand, 1 << cids, 0))
  trumps_mask = jnp.sum(jnp.where(trumps_in_hand, 1 << cids, 0))

  has_beating = beating_mask > 0
  has_same_suit = same_suit_mask > 0
  has_trumps = trumps_mask > 0

  return jax.lax.cond(
      has_beating,
      lambda: beating_mask,
      lambda: jax.lax.cond(
          has_same_suit,
          lambda: same_suit_mask,
          lambda: jax.lax.cond(
              has_trumps, lambda: trumps_mask, lambda: hand_mask
          ),
      ),
  )


def _legal_actions(state: State) -> jnp.ndarray:
  """Returns a boolean mask of legal actions."""
  actions = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_)
  me = state.current_player
  hand_mask = state.hand_masks[me]

  can_exchange = (
      ~state.closed
      & ~_talon_empty(state)
      & ~state.trump_taken
      & (state.current_player == me)
      & (state.leader == me)
      & (state.trick_cards[0] == -1)
      & (_stock_remaining(state) > 0)
      & mask_contains(hand_mask, card_id(state.trump, TRUMP_JACK_RANK))
  )
  actions = actions.at[EXCHANGE_TRUMP_ACTION].set(can_exchange)

  can_close = (
      ~state.closed
      & ~_talon_empty(state)
      & (_stock_remaining(state) >= CLOSE_MIN_ABOVE_TRUMP)
      & (state.leader == me)
      & (state.trick_cards[0] == -1)
  )
  actions = actions.at[CLOSE_TALON_ACTION].set(can_close)

  def lead_actions():
    cids = jnp.arange(NUM_CARDS)
    return (hand_mask & (1 << cids)) > 0

  def reply_actions():
    lead_cid = state.trick_cards[0]
    strict = _strict_rules_active(state)
    legal_mask = jax.lax.cond(
        strict,
        lambda: _legal_reply_mask(hand_mask, lead_cid, state.trump),
        lambda: hand_mask,
    )
    cids = jnp.arange(NUM_CARDS)
    return (legal_mask & (1 << cids)) > 0

  is_leader = (state.leader == me) & (state.trick_cards[0] == -1)
  card_actions = jnp.where(
      is_leader, lead_actions(), reply_actions()
  )
  actions = actions.at[:NUM_CARDS].set(card_actions)

  return actions


def _finalize_game(
    state: State, winner: int, cause: int
) -> State:
  """Finalizes the game and computes game points."""
  loser = 1 - winner
  game_points = jax.lax.switch(
      cause,
      [
          lambda: jnp.where(
              state.tricks_won[loser] == 0,
              3,
              jnp.where(state.points[loser] < 33, 2, 1),
          ),
          lambda: jnp.array(2),
          lambda: jnp.array(3),
          lambda: jnp.where(state.points[loser] >= 33, 1, 2),
      ],
  )
  gp = jnp.zeros(2, dtype=jnp.int32)
  gp = gp.at[winner].set(game_points)

  return state._replace(
      terminal=jnp.array(True),
      winner=jnp.array(winner),
      game_points=gp,
      current_player=jnp.array(-1),
      last_trick_winner=jnp.where(
          state.last_trick_winner == -1, winner, state.last_trick_winner
      ),
  )


def _step(state: State, action: jnp.ndarray) -> State:
  """Applies an action to the game state."""

  def _exchange_trump_jack(st: State) -> State:
    player = st.current_player
    jack_cid = card_id(st.trump, TRUMP_JACK_RANK)
    hand_mask = st.hand_masks[player] & ~(1 << jack_cid)
    hand_mask |= 1 << st.trump_card
    hand_masks = st.hand_masks.at[player].set(hand_mask)
    current_hand = st.hands[player]
    new_hand = jnp.where(current_hand == jack_cid, st.trump_card, current_hand)
    new_hand = jnp.sort(new_hand)
    hands = st.hands.at[player].set(new_hand)
    return st._replace(hand_masks=hand_masks, hands=hands, trump_card=jack_cid)

  def _apply_close(st: State) -> State:
    return st._replace(
        closed=jnp.array(True),
        closed_by=st.current_player,
        stock_idx=jnp.array(len(st.stock)),
        trump_taken=jnp.array(True),
    )

  def _play_card_action(st: State, cid: int) -> State:
    """Applies a card play action to the game state."""
    me = st.current_player

    def _score_marriage(s: State) -> State:
      r = cid_rank(cid)
      s_cid = cid_suit(cid)
      counterpart_cid = card_id(s_cid, 5 - r)
      can_score = (
          ~_strict_rules_active(s)
          & (s.trick_cards[0] == -1)
          & ((r == 2) | (r == 3))
          & ~s.marriages_scored[me, s_cid]
          & mask_contains(s.hand_masks[me], counterpart_cid)
      )
      bonus = jnp.where(s_cid == s.trump, 40, 20)
      points = s.points.at[me].add(bonus * can_score)
      marriages_scored = s.marriages_scored.at[me, s_cid].set(
          s.marriages_scored[me, s_cid] | can_score
      )
      s = s._replace(points=points, marriages_scored=marriages_scored)
      winner = jnp.where(s.points[0] >= 66, 0, 1)
      return jax.lax.cond(
          jnp.any(s.points >= 66),
          lambda: _finalize_game(s, winner, 0),
          lambda: s,
      )

    st = jax.lax.cond(
        (st.leader == me) & (st.trick_cards[0] == -1),
        _score_marriage,
        lambda s: s,
        st,
    )
    hand_mask = st.hand_masks[me] & ~(1 << cid)
    st = st._replace(hand_masks=st.hand_masks.at[me].set(hand_mask))
    current_hand = st.hands[me]
    new_hand = jnp.where(current_hand == cid, -1, current_hand)
    hands = st.hands.at[me].set(new_hand)
    st = st._replace(hands=hands)

    def _finish_trick(s: State) -> State:
      lead_card, _ = s.trick_cards
      reply_card = cid
      w_rel = _trick_winner(lead_card, reply_card, s.trump)
      w = (s.leader + w_rel) % 2
      pts = card_points(lead_card) + card_points(reply_card)
      s = s._replace(
          points=s.points.at[w].add(pts),
          tricks_won=s.tricks_won.at[w].add(1),
          last_trick_winner=w,
          trick_cards=jnp.full(2, -1, dtype=jnp.int32),
      )

      def _draw_cards(s_draw: State) -> State:
        def _add_card(hand, card):
          idx = jnp.argmax(hand == -1)
          return jnp.sort(hand.at[idx].set(card))

        def _draw_two(s2: State) -> State:
          c1 = s2.stock[s2.stock_idx]
          c2 = s2.stock[s2.stock_idx + 1]
          h_w_mask = s2.hand_masks[w] | (1 << c1)
          h_l_mask = s2.hand_masks[1 - w] | (1 << c2)
          hm = s2.hand_masks.at[w].set(h_w_mask).at[1 - w].set(h_l_mask)

          h_w = _add_card(s2.hands[w], c1)
          h_l = _add_card(s2.hands[1 - w], c2)
          hands = s2.hands.at[w].set(h_w).at[1 - w].set(h_l)
          return s2._replace(
              hand_masks=hm, hands=hands, stock_idx=s2.stock_idx + 2
          )

        def _draw_one(s1: State) -> State:
          c = s1.stock[s1.stock_idx]
          h_w_mask = s1.hand_masks[w] | (1 << c)
          h_l_mask = s1.hand_masks[1 - w] | (1 << s1.trump_card)
          hm = s1.hand_masks.at[w].set(h_w_mask).at[1 - w].set(h_l_mask)

          h_w = _add_card(s1.hands[w], c)
          h_l = _add_card(s1.hands[1 - w], s1.trump_card)
          hands = s1.hands.at[w].set(h_w).at[1 - w].set(h_l)
          return s1._replace(
              hand_masks=hm,
              hands=hands,
              stock_idx=s1.stock_idx + 1,
              trump_taken=jnp.array(True),
          )

        def _draw_last(sl: State) -> State:
          h_w_mask = sl.hand_masks[w] | (1 << sl.trump_card)
          h_w = _add_card(sl.hands[w], sl.trump_card)
          hands = sl.hands.at[w].set(h_w)
          return sl._replace(
              hand_masks=sl.hand_masks.at[w].set(h_w_mask),
              hands=hands,
              trump_taken=jnp.array(True),
          )

        return jax.lax.cond(
            _stock_remaining(s_draw) >= 2,
            _draw_two,
            lambda s_l: jax.lax.cond(
                _stock_remaining(s_l) == 1,
                _draw_one,
                lambda s_: jax.lax.cond(
                    ~s_.trump_taken, _draw_last, lambda s__: s__, s_
                ),
                s_l,
            ),
            s_draw,
        )

      s = jax.lax.cond(
          ~s.closed & ~_talon_empty(s), _draw_cards, lambda s_: s_, s
      )
      s = s._replace(leader=w, current_player=w)

      def _resolve_last_trick_winner(s_res: State) -> tuple[int, int]:
        return jax.lax.cond(
            (s_res.closed_by != -1) & (s_res.points[s_res.closed_by] < 66),
            lambda: (1 - s_res.closed_by, 1),
            lambda: (w, 3),
        )

      def _check_game_end(s_end: State) -> State:
        winner_66 = jnp.where(s_end.points[0] >= 66, 0, 1)
        s_end = jax.lax.cond(
            jnp.any(s_end.points >= 66),
            lambda: _finalize_game(s_end, winner_66, 0),
            lambda: s_end,
        )
        s_end = jax.lax.cond(
            s_end.tricks_won[w] == TRICKS_PER_GAME,
            lambda: _finalize_game(s_end, w, 2),
            lambda: s_end,
        )
        is_last_trick = jnp.sum(s_end.hand_masks) == 0
        s_end = jax.lax.cond(
            is_last_trick,
            lambda: _finalize_game(s_end, *_resolve_last_trick_winner(s_end)),
            lambda: s_end,
        )
        return s_end

      return _check_game_end(s)

    def _leader_play(s: State, card_id: int) -> State:
      return s._replace(
          trick_cards=s.trick_cards.at[0].set(card_id),
          current_player=1 - s.current_player,
      )

    def _reply_play(s: State, card_id: int) -> State:
      s = s._replace(
          trick_cards=s.trick_cards.at[1].set(card_id), current_player=-1
      )
      return jax.lax.cond(s.terminal, lambda: s, lambda: _finish_trick(s))

    return jax.lax.cond(
        (st.leader == me) & (st.trick_cards[0] == -1),
        lambda s: _leader_play(s, cid),
        lambda s: _reply_play(s, cid),
        st,
    )

  def _do_step(s):
    def _play_card_branch(op):
      state_op, act_op = op
      return _play_card_action(state_op, act_op)

    def _special_action_branch(op):
      state_op, act_op = op
      return jax.lax.switch(
          act_op - NUM_CARDS, [_exchange_trump_jack, _apply_close], state_op
      )

    return jax.lax.cond(
        action < NUM_CARDS,
        _play_card_branch,
        _special_action_branch,
        (s, action),
    )

  return jax.lax.cond(state.terminal, lambda s: s, _do_step, state)

class Snapszer:
    def __init__(self):
        self._step = jax.jit(partial(_step))
        self._legal_actions = jax.jit(_legal_actions)

    def new_initial_state(self, key: jax.random.PRNGKey) -> State:
        return State.new(key)

    def step(self, state: State, action: jnp.ndarray) -> State:
        return self._step(state, action)

    def legal_actions(self, state: State) -> jnp.ndarray:
        return self._legal_actions(state)

    def is_terminal(self, state: State) -> jnp.ndarray:
        return state.terminal

    def returns(self, state: State) -> jnp.ndarray:
        return state.game_points

    def num_actions(self) -> int:
        return TOTAL_ACTIONS
