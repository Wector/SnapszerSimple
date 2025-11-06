"""Optimized JAX implementation of Hungarian Snapszer.

Key optimizations:
1. JAX native RNG (much faster than MT19937)
2. Unsorted hands (faster insert/remove operations)
3. Reduced branching (fewer jax.lax.cond calls)
4. Streamlined state updates

Note: This version does NOT maintain parity with the base implementation
due to different RNG and unsorted hands. Use snapszer_jax.py for parity testing.
"""

from __future__ import annotations
from typing import Tuple, NamedTuple
import jax
import jax.numpy as jnp
from functools import partial

# Game constants
SUITS = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
RANKS = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
RANK_STRENGTH = jnp.array([5, 4, 3, 2, 1], dtype=jnp.int32)
CARD_POINTS = jnp.array([11, 10, 4, 3, 2], dtype=jnp.int32)
NUM_RANKS = 5
NUM_SUITS = 4
NUM_CARDS = 20
TRICKS_PER_GAME = 10
TRUMP_JACK_RANK = 4
EXCHANGE_TRUMP_ACTION = 20
CLOSE_TALON_ACTION = 21
TOTAL_ACTIONS = 22
CLOSE_MIN_ABOVE_TRUMP = 3
OBSERVATION_SIZE = 101  # 20+4+20+3+2+2+20+20+2+4+4
MAX_HAND_SIZE = 10


# Helper functions
def card_id(suit: int, rank: int) -> int:
    """Convert suit and rank to card ID."""
    return suit * NUM_RANKS + rank


@jax.jit
def cid_suit(cid: jnp.ndarray) -> jnp.ndarray:
    """Extract suit from card ID."""
    return cid // NUM_RANKS


@jax.jit
def cid_rank(cid: jnp.ndarray) -> jnp.ndarray:
    """Extract rank from card ID."""
    return cid % NUM_RANKS


@jax.jit
def card_points_fn(cid: jnp.ndarray) -> jnp.ndarray:
    """Get point value of a card."""
    return CARD_POINTS[cid_rank(cid)]


@jax.jit
def mask_contains(mask: jnp.ndarray, cid: jnp.ndarray) -> jnp.ndarray:
    """Check if bitmask contains a card."""
    return ((mask >> cid) & 1) == 1


class SnapszerState(NamedTuple):
    """Optimized immutable game state."""
    key: jax.Array  # JAX RNG key
    trump: jnp.int32
    trump_card: jnp.int32
    stock: jax.Array  # [9] - draw pile
    stock_idx: jnp.int32
    hands: jax.Array  # [2, MAX_HAND_SIZE] - UNSORTED, padded with -1
    hand_sizes: jax.Array  # [2]
    hand_masks: jax.Array  # [2] - bitmasks for fast lookup
    current_player: jnp.int32
    leader: jnp.int32
    trick_cards: jax.Array  # [2]
    points: jax.Array  # [2]
    tricks_won: jax.Array  # [2]
    closed: jnp.bool_
    closed_by: jnp.int32
    trump_taken: jnp.bool_
    last_trick_winner: jnp.int32
    marriages_scored: jax.Array  # [2, 4]
    terminal: jnp.bool_
    winner: jnp.int32
    game_points: jax.Array  # [2]


@jax.jit
def new_game(key: jax.Array) -> SnapszerState:
    """Create new game with JAX RNG key."""
    # Shuffle deck using JAX RNG (much faster than MT19937)
    deck = jax.random.permutation(key, jnp.arange(NUM_CARDS, dtype=jnp.int32))

    # Trump card and suit
    trump_card = deck[10]
    trump = cid_suit(trump_card)

    # Deal hands (unsorted for speed)
    p0_hand = jnp.concatenate([deck[:5], jnp.full(5, -1, dtype=jnp.int32)])
    p1_hand = jnp.concatenate([deck[5:10], jnp.full(5, -1, dtype=jnp.int32)])
    hands = jnp.stack([p0_hand, p1_hand])
    hand_sizes = jnp.array([5, 5], dtype=jnp.int32)

    # Hand masks for fast membership testing
    def make_mask(hand):
        return jnp.bitwise_or.reduce(jnp.where(hand >= 0, 1 << hand, 0))
    hand_masks = jax.vmap(make_mask)(hands)

    # Stock (remaining cards)
    stock = deck[11:]

    # Initial player: non-dealer (player 1)
    initial_player = jnp.int32(1)

    return SnapszerState(
        key=key,
        trump=trump,
        trump_card=trump_card,
        stock=stock,
        stock_idx=jnp.int32(0),
        hands=hands,
        hand_sizes=hand_sizes,
        hand_masks=hand_masks,
        current_player=initial_player,
        leader=initial_player,
        trick_cards=jnp.array([-1, -1], dtype=jnp.int32),
        points=jnp.zeros(2, dtype=jnp.int32),
        tricks_won=jnp.zeros(2, dtype=jnp.int32),
        closed=jnp.bool_(False),
        closed_by=jnp.int32(-1),
        trump_taken=jnp.bool_(False),
        last_trick_winner=jnp.int32(-1),
        marriages_scored=jnp.zeros((2, 4), dtype=jnp.bool_),
        terminal=jnp.bool_(False),
        winner=jnp.int32(-1),
        game_points=jnp.zeros(2, dtype=jnp.int32)
    )


# OPTIMIZED: Fast unsorted hand operations
@jax.jit
def remove_card_from_hand(state: SnapszerState, player: jnp.ndarray, cid: jnp.ndarray) -> SnapszerState:
    """Remove card from UNSORTED hand - much faster than sorted version."""
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]

    # Find card position
    matches = (hand == cid) & (jnp.arange(MAX_HAND_SIZE) < hand_size)
    card_idx = jnp.argmax(matches)  # First occurrence
    has_match = jnp.any(matches)

    # Swap-and-pop: move last card to removed position
    last_idx = hand_size - 1
    last_card = hand[last_idx]

    # Build new hand using vectorized operations
    new_hand = jnp.where(
        jnp.arange(MAX_HAND_SIZE) == card_idx,
        jnp.where(card_idx == last_idx, -1, last_card),  # If removing last, just set -1
        jnp.where(jnp.arange(MAX_HAND_SIZE) == last_idx, -1, hand)  # Clear last position
    )

    # Update state (use jnp.where for branchless updates)
    final_hand = jnp.where(has_match, new_hand, hand)
    final_size = jnp.where(has_match, hand_size - 1, hand_size)
    final_mask = jnp.where(has_match, state.hand_masks[player] & ~(1 << cid), state.hand_masks[player])

    return state._replace(
        hands=state.hands.at[player].set(final_hand),
        hand_sizes=state.hand_sizes.at[player].set(final_size),
        hand_masks=state.hand_masks.at[player].set(final_mask)
    )


@jax.jit
def insert_card_to_hand(state: SnapszerState, player: jnp.ndarray, cid: jnp.ndarray) -> SnapszerState:
    """Insert card to UNSORTED hand - O(1) instead of O(n)."""
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]

    # Simply append to end (no sorting needed)
    new_hand = hand.at[hand_size].set(cid)
    new_size = hand_size + 1
    new_mask = state.hand_masks[player] | (1 << cid)

    return state._replace(
        hands=state.hands.at[player].set(new_hand),
        hand_sizes=state.hand_sizes.at[player].set(new_size),
        hand_masks=state.hand_masks.at[player].set(new_mask)
    )


# Game logic helper functions
@jax.jit
def trick_winner(lead_cid: jnp.ndarray, reply_cid: jnp.ndarray, trump: jnp.ndarray) -> jnp.ndarray:
    """Determine trick winner. Returns 0 if leader wins, 1 if follower wins."""
    lead_suit = cid_suit(lead_cid)
    lead_rank = cid_rank(lead_cid)
    reply_suit = cid_suit(reply_cid)
    reply_rank = cid_rank(reply_cid)

    # Reply is trump and lead is not
    reply_trumps = (reply_suit == trump) & (lead_suit != trump)

    # Same suit and reply is stronger
    same_suit = reply_suit == lead_suit
    reply_stronger = RANK_STRENGTH[reply_rank] > RANK_STRENGTH[lead_rank]
    reply_beats_same_suit = same_suit & reply_stronger

    return jnp.where(reply_trumps | reply_beats_same_suit, jnp.int32(1), jnp.int32(0))


@jax.jit
def strict_rules_active(state: SnapszerState) -> jnp.ndarray:
    """Check if strict follow rules are active."""
    return state.closed | (state.trump_taken & (state.stock_idx >= len(state.stock)))


@jax.jit
def can_exchange_trump_jack(state: SnapszerState, player: jnp.ndarray) -> jnp.ndarray:
    """Check if player can exchange trump jack."""
    jack_cid = state.trump * NUM_RANKS + TRUMP_JACK_RANK
    has_jack = mask_contains(state.hand_masks[player], jack_cid)

    # Count cards above trump card
    trump_value = cid_rank(state.trump_card)
    trump_suit = state.trump

    def count_higher(i):
        card = state.hands[player, i]
        is_valid = (i < state.hand_sizes[player]) & (card >= 0)
        is_trump_suit = cid_suit(card) == trump_suit
        is_higher = cid_rank(card) < trump_value  # Lower rank number = higher value
        return is_valid & is_trump_suit & is_higher

    higher_count = jnp.sum(jax.vmap(count_higher)(jnp.arange(MAX_HAND_SIZE)))

    return (
        has_jack &
        ~state.closed &
        ~state.trump_taken &
        (state.trick_cards[0] == -1) &
        (state.leader == player) &
        (state.stock_idx < len(state.stock))
    )


@jax.jit
def legal_reply_cards_bitmask(hand_mask: jnp.ndarray, hand_size: jnp.ndarray, lead_cid: jnp.ndarray, trump: jnp.ndarray) -> jnp.ndarray:
    """Get legal reply cards as a bitmask. ULTRA-OPTIMIZED."""
    lead_s = cid_suit(lead_cid)
    lead_r = cid_rank(lead_cid)
    lead_strength = RANK_STRENGTH[lead_r]

    all_cids = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    card_suits = jax.vmap(cid_suit)(all_cids)
    card_ranks = jax.vmap(cid_rank)(all_cids)
    card_strengths = RANK_STRENGTH[card_ranks]

    has_card = jax.vmap(lambda cid: mask_contains(hand_mask, cid))(all_cids)
    same_suit = card_suits == lead_s
    beating = same_suit & (card_strengths > lead_strength)
    is_trump = card_suits == trump

    # Count what we have
    has_same_suit = jnp.any(has_card & same_suit)
    has_beating = jnp.any(has_card & beating)
    has_trumps = jnp.any(has_card & is_trump)

    # Determine legal cards using select
    legal = jnp.select(
        [has_same_suit & has_beating, has_same_suit & ~has_beating, has_trumps],
        [beating, same_suit, is_trump],
        default=has_card
    )

    # Convert to bitmask directly
    legal_bitmask = jnp.bitwise_or.reduce(jnp.where(legal, 1 << all_cids, 0))

    return legal_bitmask


@jax.jit
def legal_actions_mask(state: SnapszerState) -> jax.Array:
    """Get legal actions as boolean mask [TOTAL_ACTIONS]."""
    mask = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_)

    # If terminal, no actions
    mask = jnp.where(state.terminal, mask, mask)

    me = state.current_player
    hand_mask = state.hand_masks[me]

    # Exchange trump jack
    can_exchange = can_exchange_trump_jack(state, me)
    mask = mask.at[EXCHANGE_TRUMP_ACTION].set(can_exchange)

    # Close talon
    can_close = (
        (~state.closed) &
        (~state.trump_taken) &
        ((len(state.stock) - state.stock_idx) >= CLOSE_MIN_ABOVE_TRUMP) &
        (state.leader == me) &
        (state.trick_cards[0] == -1)
    )
    mask = mask.at[CLOSE_TALON_ACTION].set(can_close)

    # Card plays
    is_leader = (state.leader == me) & (state.trick_cards[0] == -1)
    is_replier = state.trick_cards[0] >= 0

    # OPTIMIZED: Use bitmask directly for card actions
    # Leader can play any card in hand
    leader_cards_mask = jnp.where(is_leader, hand_mask, 0)

    # Replier must follow rules
    strict_active = strict_rules_active(state)
    replier_cards_mask = jnp.where(
        is_replier,
        jnp.where(
            strict_active,
            hand_mask & legal_reply_cards_bitmask(hand_mask, state.hand_sizes[me], state.trick_cards[0], state.trump),
            hand_mask
        ),
        0
    )

    # Combine leader and replier masks
    cards_mask = leader_cards_mask | replier_cards_mask

    # Set card action bits
    card_actions = (cards_mask >> jnp.arange(NUM_CARDS, dtype=jnp.int32)) & 1
    mask = mask.at[:NUM_CARDS].set(card_actions.astype(jnp.bool_))

    return mask


# Scoring and game-ending logic
@jax.jit
def compute_game_points(state: SnapszerState, winner: jnp.ndarray, cause: jnp.ndarray) -> jax.Array:
    """Compute game points. Cause: 0=closer_fail, 1=durchmarsch, 2=auto66, 3=last_trick."""
    loser = 1 - winner

    gp = jnp.where(
        cause == 0, jnp.int32(2),  # closer_fail
        jnp.where(
            cause == 1, jnp.int32(3),  # durchmarsch
            jnp.where(
                cause == 2,  # auto66
                jnp.where(
                    state.tricks_won[loser] == 0, jnp.int32(3),
                    jnp.where(state.points[loser] < 33, jnp.int32(2), jnp.int32(1))
                ),
                jnp.where(state.points[loser] >= 33, jnp.int32(1), jnp.int32(2))  # last_trick
            )
        )
    )

    result = jnp.zeros(2, dtype=jnp.int32)
    return result.at[winner].set(gp)


@jax.jit
def finalize_game(state: SnapszerState, winner: jnp.ndarray, cause: jnp.ndarray) -> SnapszerState:
    """Finalize game (branchless version)."""
    should_finalize = ~state.terminal

    new_winner = jnp.where(should_finalize, winner, state.winner)
    new_last_trick = jnp.where(should_finalize & (state.last_trick_winner < 0), winner, state.last_trick_winner)
    new_game_points = jnp.where(
        should_finalize,
        compute_game_points(state, winner, cause),
        state.game_points
    )

    return state._replace(
        terminal=state.terminal | should_finalize,
        winner=new_winner,
        last_trick_winner=new_last_trick,
        current_player=jnp.where(should_finalize, jnp.int32(-1), state.current_player),
        game_points=new_game_points
    )


@jax.jit
def maybe_finalize_on_66(state: SnapszerState) -> SnapszerState:
    """Check for 66 points and finalize."""
    has_winner = (state.points[0] >= 66) | (state.points[1] >= 66)
    winner = jnp.where(state.points[0] >= 66, jnp.int32(0), jnp.int32(1))
    return jax.lax.cond(
        has_winner & ~state.terminal,
        lambda s: finalize_game(s, winner, jnp.int32(2)),
        lambda s: s,
        state
    )


@jax.jit
def score_marriage_if_any(state: SnapszerState, player: jnp.ndarray, played_cid: jnp.ndarray) -> SnapszerState:
    """Score marriage (branchless version)."""
    r = cid_rank(played_cid)
    s = cid_suit(played_cid)

    is_king_or_queen = (r == 2) | (r == 3)
    counterpart_rank = jnp.where(r == 2, jnp.int32(3), jnp.int32(2))
    counterpart_cid = s * NUM_RANKS + counterpart_rank

    should_score = (
        (~strict_rules_active(state)) &
        (state.trick_cards[0] == -1) &
        is_king_or_queen &
        ~state.marriages_scored[player, s] &
        mask_contains(state.hand_masks[player], counterpart_cid)
    )

    bonus = jnp.where(s == state.trump, 40, 20)

    new_marriages = state.marriages_scored.at[player, s].set(
        state.marriages_scored[player, s] | should_score
    )
    new_points = state.points.at[player].add(jnp.where(should_score, bonus, 0))

    state = state._replace(marriages_scored=new_marriages, points=new_points)
    return maybe_finalize_on_66(state)


@jax.jit
def finish_trick(state: SnapszerState) -> SnapszerState:
    """Finish trick and handle drawing."""
    lead_card = state.trick_cards[0]
    reply_card = state.trick_cards[1]

    w_rel = trick_winner(lead_card, reply_card, state.trump)
    w = (state.leader + w_rel) % 2

    pts = card_points_fn(lead_card) + card_points_fn(reply_card)
    state = state._replace(
        points=state.points.at[w].add(pts),
        tricks_won=state.tricks_won.at[w].add(1),
        last_trick_winner=w,
        trick_cards=jnp.array([-1, -1], dtype=jnp.int32)
    )

    state = maybe_finalize_on_66(state)

    # Handle card drawing (only if not terminal)
    return jax.lax.cond(state.terminal, lambda s: s, lambda s: _handle_draw(s, w), state)


@jax.jit
def _handle_draw(state: SnapszerState, w: jnp.ndarray) -> SnapszerState:
    """Handle card drawing after trick."""
    should_draw = (~state.closed) & (state.stock_idx < len(state.stock))

    def draw_cards(s):
        remaining = len(s.stock) - s.stock_idx

        def draw_two(s):
            card1 = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, w, card1)
            s = s._replace(stock_idx=s.stock_idx + 1)
            card2 = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, 1 - w, card2)
            return s._replace(stock_idx=s.stock_idx + 1)

        def draw_one(s):
            card = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, w, card)
            s = s._replace(stock_idx=s.stock_idx + 1)
            return jax.lax.cond(
                ~s.trump_taken,
                lambda s: insert_card_to_hand(s, 1 - w, s.trump_card)._replace(trump_taken=jnp.bool_(True)),
                lambda s: s,
                s
            )

        def draw_trump_only(s):
            return jax.lax.cond(
                ~s.trump_taken,
                lambda s: insert_card_to_hand(s, w, s.trump_card)._replace(trump_taken=jnp.bool_(True)),
                lambda s: s,
                s
            )

        return jax.lax.cond(
            remaining >= 2, draw_two,
            lambda s: jax.lax.cond(remaining == 1, draw_one, draw_trump_only, s),
            s
        )

    state = jax.lax.cond(should_draw, draw_cards, lambda s: s, state)
    state = state._replace(leader=w, current_player=w)

    # Check durchmarsch
    state = jax.lax.cond(
        state.tricks_won[w] == TRICKS_PER_GAME,
        lambda s: finalize_game(s, w, jnp.int32(1)),
        lambda s: s,
        state
    )

    # Check game end
    both_empty = (state.hand_sizes[0] == 0) & (state.hand_sizes[1] == 0)

    def resolve_last(s):
        closer_failed = (s.closed_by >= 0) & (s.points[s.closed_by] < 66)
        final_winner = jnp.where(closer_failed, 1 - s.closed_by, w)
        cause = jnp.where(closer_failed, jnp.int32(0), jnp.int32(3))
        return finalize_game(s, final_winner, cause)

    return jax.lax.cond(both_empty & ~state.terminal, resolve_last, lambda s: s, state)


# Action handlers
@jax.jit
def exchange_trump_jack(state: SnapszerState, player: jnp.ndarray) -> SnapszerState:
    """Exchange trump jack."""
    jack_cid = state.trump * NUM_RANKS + TRUMP_JACK_RANK
    state = remove_card_from_hand(state, player, jack_cid)
    state = insert_card_to_hand(state, player, state.trump_card)
    return state._replace(trump_card=jack_cid)


@jax.jit
def apply_close(state: SnapszerState) -> SnapszerState:
    """Close talon."""
    return state._replace(
        closed=jnp.bool_(True),
        closed_by=state.current_player,
        stock_idx=jnp.int32(len(state.stock)),
        trump_taken=jnp.bool_(True)
    )


@jax.jit
def apply_action(state: SnapszerState, action: jnp.ndarray) -> SnapszerState:
    """Apply action to state."""
    return jax.lax.cond(state.terminal, lambda s: s, lambda s: _apply_action_impl(s, action), state)


@jax.jit
def _apply_action_impl(state: SnapszerState, action: jnp.ndarray) -> SnapszerState:
    """Internal action implementation."""
    me = state.current_player

    def handle_exchange(s):
        return exchange_trump_jack(s, me)

    def handle_close(s):
        return apply_close(s)

    def handle_card_play(s):
        cid = action
        is_leading = (s.leader == me) & (s.trick_cards[0] == -1)

        def lead_card(s):
            s = score_marriage_if_any(s, me, cid)

            def continue_lead(s):
                s = s._replace(trick_cards=s.trick_cards.at[0].set(cid))
                s = remove_card_from_hand(s, me, cid)
                return s._replace(current_player=1 - me)

            return jax.lax.cond(s.terminal, lambda s: s, continue_lead, s)

        def reply_card(s):
            s = s._replace(trick_cards=s.trick_cards.at[1].set(cid))
            s = remove_card_from_hand(s, me, cid)
            return finish_trick(s)

        return jax.lax.cond(is_leading, lead_card, reply_card, s)

    return jax.lax.cond(
        action == EXCHANGE_TRUMP_ACTION, handle_exchange,
        lambda s: jax.lax.cond(action == CLOSE_TALON_ACTION, handle_close, handle_card_play, s),
        state
    )


def returns(state: SnapszerState) -> Tuple[float, float]:
    """Get returns for both players."""
    if not state.terminal or state.winner == -1:
        return (0.0, 0.0)
    diff = state.game_points[0] - state.game_points[1]
    return (float(diff), float(-diff))


@jax.jit
def observation_tensor(state: SnapszerState, player: int) -> jax.Array:
    """
    Create observation tensor for player.

    Encodes all information visible to the player:
    - Own hand (20 bits as one-hot)
    - Trump suit (4 bits one-hot)
    - Trump card (20 bits one-hot, or all zeros if taken)
    - Public state (stock_idx, closed, trump_taken, points, tricks_won, etc.)
    - Trick cards
    - Marriages scored

    Returns flat array of features.
    """
    player = jnp.int32(player)
    opponent = 1 - player

    # Own hand (20 bits) - extract bits from bitmask using vectorized operations
    card_indices = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    hand_bits = ((state.hand_masks[player] >> card_indices) & 1).astype(jnp.float32)

    # Trump suit (4 bits one-hot) - vectorized comparison
    suit_indices = jnp.arange(NUM_SUITS, dtype=jnp.int32)
    trump_bits = (state.trump == suit_indices).astype(jnp.float32)

    # Trump card (20 bits one-hot, or all zeros if taken) - vectorized
    trump_card_one_hot = (state.trump_card == card_indices).astype(jnp.float32)
    trump_card_bits = jnp.where(state.trump_taken, 0.0, trump_card_one_hot)

    # Scalar features
    stock_cards_left = jnp.float32(9 - state.stock_idx)  # stock has 9 cards
    closed_bit = jnp.float32(state.closed)
    trump_taken_bit = jnp.float32(state.trump_taken)

    # Points (normalized)
    my_points = jnp.float32(state.points[player]) / 120.0
    opp_points = jnp.float32(state.points[opponent]) / 120.0

    # Tricks won
    my_tricks = jnp.float32(state.tricks_won[player]) / 10.0
    opp_tricks = jnp.float32(state.tricks_won[opponent]) / 10.0

    # Trick cards (2 cards, 20 bits each one-hot) - vectorized
    trick_card_0_bits = (state.trick_cards[0] == card_indices).astype(jnp.float32)
    trick_card_1_bits = (state.trick_cards[1] == card_indices).astype(jnp.float32)

    # Current player and leader
    is_my_turn = jnp.float32(state.current_player == player)
    is_leader = jnp.float32(state.leader == player)

    # Marriages scored (2 players × 4 suits) - already arrays
    my_marriages = state.marriages_scored[player].astype(jnp.float32)
    opp_marriages = state.marriages_scored[opponent].astype(jnp.float32)

    # Concatenate all features
    obs = jnp.concatenate([
        hand_bits,  # 20
        trump_bits,  # 4
        trump_card_bits,  # 20
        jnp.array([stock_cards_left, closed_bit, trump_taken_bit]),  # 3
        jnp.array([my_points, opp_points]),  # 2
        jnp.array([my_tricks, opp_tricks]),  # 2
        trick_card_0_bits,  # 20
        trick_card_1_bits,  # 20
        jnp.array([is_my_turn, is_leader]),  # 2
        my_marriages,  # 4
        opp_marriages,  # 4
    ])

    return obs  # Total: 20+4+20+3+2+2+20+20+2+4+4 = 101 features
