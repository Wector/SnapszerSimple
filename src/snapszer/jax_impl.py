"""JAX implementation of Hungarian Snapszer (Snapser) card game.

This is a pure functional implementation using JAX arrays for GPU acceleration
and automatic differentiation support. It maintains parity with snapszer_base.py.
"""

from __future__ import annotations
from typing import Tuple, NamedTuple, Optional
import jax
import jax.numpy as jnp
from functools import partial

# Game constants (same as base)
SUITS = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
RANKS = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)  # A,10,K,Q,J
RANK_STRENGTH = jnp.array([5, 4, 3, 2, 1], dtype=jnp.int32)  # indexed by rank
CARD_POINTS = jnp.array([11, 10, 4, 3, 2], dtype=jnp.int32)  # indexed by rank
NUM_RANKS = 5
NUM_SUITS = 4
NUM_CARDS = 20
TRICKS_PER_GAME = 10
TRUMP_JACK_RANK = 4
EXCHANGE_TRUMP_ACTION = 20
CLOSE_TALON_ACTION = 21
TOTAL_ACTIONS = 22
CLOSE_MIN_ABOVE_TRUMP = 3
OBSERVATION_SIZE = 80
MAX_HAND_SIZE = 10  # Maximum cards a player can hold


# Helper functions
def card_id(suit: int, rank: int) -> int:
    """Convert suit and rank to card ID."""
    return suit * NUM_RANKS + rank


def cid_suit(cid: jnp.ndarray) -> jnp.ndarray:
    """Extract suit from card ID."""
    return cid // NUM_RANKS


def cid_rank(cid: jnp.ndarray) -> jnp.ndarray:
    """Extract rank from card ID."""
    return cid % NUM_RANKS


def card_points_fn(cid: jnp.ndarray) -> jnp.ndarray:
    """Get point value of a card."""
    return CARD_POINTS[cid_rank(cid)]


def cards_to_mask(cards: jnp.ndarray) -> jnp.ndarray:
    """Convert card array to bitmask. Cards array should be padded with -1."""
    # Create a bitmask from valid cards (not -1)
    valid_mask = cards >= 0
    shifts = jnp.where(valid_mask, 1 << cards, 0)
    return jnp.bitwise_or.reduce(shifts)


def mask_contains(mask: jnp.ndarray, cid: jnp.ndarray) -> jnp.ndarray:
    """Check if bitmask contains a card."""
    return ((mask >> cid) & 1) == 1


class SnapszerState(NamedTuple):
    """Immutable game state for JAX."""
    # RNG key
    key: jax.Array

    # Deck and stock
    trump: jnp.int32
    trump_card: jnp.int32
    deck: jax.Array  # [20] - initial shuffled deck
    stock: jax.Array  # [9] - draw pile (deck[11:])
    stock_idx: jnp.int32

    # Hands (fixed size arrays, padded with -1)
    hands: jax.Array  # [2, MAX_HAND_SIZE] - each player's cards
    hand_sizes: jax.Array  # [2] - actual number of cards
    hand_masks: jax.Array  # [2] - bitmasks for fast lookup

    # Turn state
    current_player: jnp.int32
    leader: jnp.int32
    trick_cards: jax.Array  # [2] - current trick, -1 means None

    # Score tracking
    points: jax.Array  # [2]
    tricks_won: jax.Array  # [2]

    # Special states
    closed: jnp.bool_
    closed_by: jnp.int32  # -1 means None
    trump_taken: jnp.bool_
    last_trick_winner: jnp.int32  # -1 means None
    marriages_scored: jax.Array  # [2, 4] - bool array for each player/suit

    # Terminal state
    terminal: jnp.bool_
    winner: jnp.int32  # -1 means None
    game_points: jax.Array  # [2]


# MT19937 RNG implementation in JAX
@jax.jit
def mt19937_init(seed: jnp.ndarray) -> jax.Array:
    """Initialize MT19937 state array."""
    seed = jnp.uint32(seed)
    mt = jnp.zeros(624, dtype=jnp.uint32)
    mt = mt.at[0].set(seed)

    # Pre-compute all indices as uint32 array
    indices = jnp.arange(1, 624, dtype=jnp.uint32)

    def compute_mt_value(i_val, prev_val):
        """Compute next MT value given index and previous value."""
        temp = prev_val ^ (prev_val >> jnp.uint32(30))
        result = jnp.uint32(1812433253) * temp + i_val
        return result

    # Scan to compute all values
    def scan_fn(prev, i_val):
        new_val = compute_mt_value(i_val, prev)
        return new_val, new_val

    _, mt_values = jax.lax.scan(scan_fn, mt[0], indices)

    # Update mt with computed values
    mt = mt.at[1:].set(mt_values)

    return mt


@jax.jit
def mt19937_twist(mt: jax.Array) -> jax.Array:
    """Perform MT19937 twist operation."""
    def twist_step(i, mt):
        x = (mt[i] & jnp.uint32(0x80000000)) + (mt[(i + 1) % 624] & jnp.uint32(0x7FFFFFFF))
        xA = x >> jnp.uint32(1)
        xA = jnp.where(x & jnp.uint32(1), xA ^ jnp.uint32(0x9908B0DF), xA)
        new_val = mt[(i + 397) % 624] ^ xA
        return mt.at[i].set(new_val)

    return jax.lax.fori_loop(0, 624, twist_step, mt)


@jax.jit
def mt19937_extract(mt: jax.Array, index: jnp.ndarray) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """Extract a random number from MT19937 state."""
    # Twist if needed
    index = jnp.int32(index)
    mt = jax.lax.cond(index >= 624, lambda mt: mt19937_twist(mt), lambda mt: mt, mt)
    actual_idx = jnp.where(index >= 624, jnp.int32(0), index)

    y = mt[actual_idx]
    y = y ^ (y >> jnp.uint32(11))
    y = y ^ ((y << jnp.uint32(7)) & jnp.uint32(0x9D2C5680))
    y = y ^ ((y << jnp.uint32(15)) & jnp.uint32(0xEFC60000))
    y = y ^ (y >> jnp.uint32(18))

    new_index = (actual_idx + jnp.int32(1)) % jnp.int32(624)
    new_index = jnp.where(index >= 624, jnp.int32(1), new_index)

    return mt, new_index, y & jnp.uint32(0xFFFFFFFF)


@jax.jit
def mt19937_random_double(mt: jax.Array, index: jnp.ndarray) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    """Generate a random double in [0, 1) using MT19937."""
    mt, index, a_raw = mt19937_extract(mt, index)
    mt, index, b_raw = mt19937_extract(mt, index)

    a = a_raw >> jnp.uint32(5)
    b = b_raw >> jnp.uint32(6)
    result = (jnp.float64(a) * 67108864.0 + jnp.float64(b)) / 9007199254740992.0

    return mt, index, result


@jax.jit
def mt_shuffle(values: jax.Array, seed: jnp.ndarray) -> jax.Array:
    """Fisher-Yates shuffle using MT19937 RNG."""
    mt = mt19937_init(seed)
    index = jnp.int32(624)  # Start at 624 to trigger twist on first extract (like base impl)

    def shuffle_step(i, state):
        values, mt, index = state
        # Generate random index in [0, i+1)
        mt, index, rand_val = mt19937_random_double(mt, index)
        j = jnp.int32(rand_val * (i + 1))

        # Swap values[i] and values[j]
        temp = values[i]
        values = values.at[i].set(values[j])
        values = values.at[j].set(temp)

        return (values, mt, index)

    # Shuffle from end to start
    n = len(values)
    values, mt, index = jax.lax.fori_loop(
        1, n,
        lambda i, state: shuffle_step(n - i, state),
        (values, mt, index)
    )

    return values


@jax.jit
def trick_winner(lead_cid: jnp.ndarray, reply_cid: jnp.ndarray, trump: jnp.ndarray) -> jnp.ndarray:
    """Determine trick winner (0 = leader, 1 = replier)."""
    ls, lr = cid_suit(lead_cid), cid_rank(lead_cid)
    rs, rr = cid_suit(reply_cid), cid_rank(reply_cid)

    # Same suit: compare rank strength
    same_suit_winner = jnp.where(
        RANK_STRENGTH[lr] >= RANK_STRENGTH[rr],
        jnp.int32(0),
        jnp.int32(1)
    )

    # Different suit: check if reply is trump
    reply_is_trump = (rs == trump) & (ls != trump)

    # If same suit, use same_suit_winner; otherwise check trump
    return jnp.where(
        rs == ls,
        same_suit_winner,
        jnp.where(reply_is_trump, jnp.int32(1), jnp.int32(0))
    )


@jax.jit
def legal_reply_cards(hand: jax.Array, hand_size: jnp.ndarray, lead_cid: jnp.ndarray, trump: jnp.ndarray) -> jax.Array:
    """Compute legal reply cards under strict rules. Returns [MAX_HAND_SIZE] with -1 padding."""
    lead_s = cid_suit(lead_cid)
    lead_r = cid_rank(lead_cid)
    lead_strength = RANK_STRENGTH[lead_r]

    # Process all cards but mask out invalid ones
    def process_card(i):
        is_valid = i < hand_size
        card = hand[i]
        card_suit = cid_suit(card)
        card_rank = cid_rank(card)
        card_strength = RANK_STRENGTH[card_rank]

        same_suit = card_suit == lead_s
        beats = same_suit & (card_strength > lead_strength)
        is_trump = card_suit == trump

        return is_valid, card, same_suit, beats, is_trump

    indices = jnp.arange(MAX_HAND_SIZE)
    is_valid, cards, same_suit_mask, beating_mask, trump_mask = jax.vmap(process_card)(indices)

    # Count what we have
    has_same_suit = jnp.any(is_valid & same_suit_mask)
    has_beating = jnp.any(is_valid & beating_mask)
    has_trumps = jnp.any(is_valid & trump_mask)

    # Determine which cards are legal based on rules
    def compute_legal_mask():
        # If have same suit cards
        def same_suit_branch(_):
            # If have beating cards, use those; otherwise use all same suit
            return jax.lax.cond(
                has_beating,
                lambda _: is_valid & beating_mask,
                lambda _: is_valid & same_suit_mask,
                None
            )

        # If no same suit but have trumps
        def trump_branch(_):
            return jax.lax.cond(
                has_trumps,
                lambda _: is_valid & trump_mask,
                lambda _: is_valid,  # All cards legal
                None
            )

        return jax.lax.cond(has_same_suit, same_suit_branch, trump_branch, None)

    legal_mask = compute_legal_mask()

    # Build result array with -1 padding
    result = jnp.where(legal_mask, cards, -1)

    return result


@jax.jit
def new_game(seed: jnp.ndarray) -> SnapszerState:
    """Create a new game state with the given seed."""
    # Create and shuffle deck
    deck = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    deck = mt_shuffle(deck, seed)

    # Deal hands (sorted)
    p0_hand = jnp.sort(deck[:5])
    p1_hand = jnp.sort(deck[5:10])

    # Pad hands to MAX_HAND_SIZE
    hands = jnp.full((2, MAX_HAND_SIZE), -1, dtype=jnp.int32)
    hands = hands.at[0, :5].set(p0_hand)
    hands = hands.at[1, :5].set(p1_hand)

    # Trump card and suit
    trump_card = deck[10]
    trump = cid_suit(trump_card)

    # Stock (cards 11-19)
    stock = deck[11:]

    # Create initial state
    return SnapszerState(
        key=jax.random.PRNGKey(seed),
        trump=trump,
        trump_card=trump_card,
        deck=deck,
        stock=stock,
        stock_idx=jnp.int32(0),
        hands=hands,
        hand_sizes=jnp.array([5, 5], dtype=jnp.int32),
        hand_masks=jnp.array([cards_to_mask(p0_hand), cards_to_mask(p1_hand)], dtype=jnp.int32),
        current_player=jnp.int32(0),
        leader=jnp.int32(0),
        trick_cards=jnp.array([-1, -1], dtype=jnp.int32),
        points=jnp.array([0, 0], dtype=jnp.int32),
        tricks_won=jnp.array([0, 0], dtype=jnp.int32),
        closed=jnp.bool_(False),
        closed_by=jnp.int32(-1),
        trump_taken=jnp.bool_(False),
        last_trick_winner=jnp.int32(-1),
        marriages_scored=jnp.zeros((2, 4), dtype=jnp.bool_),
        terminal=jnp.bool_(False),
        winner=jnp.int32(-1),
        game_points=jnp.array([0, 0], dtype=jnp.int32),
    )


@jax.jit
def stock_remaining(state: SnapszerState) -> jnp.ndarray:
    """Number of cards remaining in stock."""
    return jnp.maximum(0, len(state.stock) - state.stock_idx)


@jax.jit
def talon_cards_remaining(state: SnapszerState) -> jnp.ndarray:
    """Total cards in talon (stock + trump card if not taken)."""
    return stock_remaining(state) + jnp.where(state.trump_taken, 0, 1)


@jax.jit
def talon_empty(state: SnapszerState) -> jnp.ndarray:
    """Check if talon is empty."""
    return state.trump_taken & (state.stock_idx >= len(state.stock))


@jax.jit
def strict_rules_active(state: SnapszerState) -> jnp.ndarray:
    """Check if strict follow rules are in effect."""
    return state.closed | talon_empty(state)


@jax.jit
def can_exchange_trump_jack(state: SnapszerState, player: jnp.ndarray) -> jnp.ndarray:
    """Check if player can exchange trump jack."""
    # Must be current player and leader
    is_valid_player = (state.current_player == player) & (state.leader == player)

    # Must be before trick started
    no_trick = state.trick_cards[0] == -1

    # Talon must be open and have cards
    talon_ok = (~state.closed) & (~state.trump_taken) & (stock_remaining(state) > 0)

    # Must have trump jack
    jack_cid = card_id(state.trump, TRUMP_JACK_RANK)
    has_jack = mask_contains(state.hand_masks[player], jack_cid)

    return is_valid_player & no_trick & talon_ok & has_jack


@jax.jit
def observation_tensor(state: SnapszerState, player: jnp.ndarray) -> jax.Array:
    """Generate observation tensor for a player."""
    obs = jnp.zeros(OBSERVATION_SIZE, dtype=jnp.float32)

    # Player's hand [0-19]
    hand_full = state.hands[player]
    def set_hand_card(i, obs_arr):
        card = hand_full[i]
        is_valid = card >= 0
        return jnp.where(is_valid, obs_arr.at[card].set(1.0), obs_arr)
    obs = jax.lax.fori_loop(0, MAX_HAND_SIZE, set_hand_card, obs)

    # Lead trick card [20-39]
    obs = jnp.where(
        state.trick_cards[0] >= 0,
        obs.at[20 + state.trick_cards[0]].set(1.0),
        obs
    )

    # Reply trick card [40-59]
    obs = jnp.where(
        state.trick_cards[1] >= 0,
        obs.at[40 + state.trick_cards[1]].set(1.0),
        obs
    )

    # Trump suit [60-63]
    obs = obs.at[60 + state.trump].set(1.0)

    # Trump taken [64]
    obs = obs.at[64].set(jnp.float32(state.trump_taken))

    # Talon closed [65]
    obs = obs.at[65].set(jnp.float32(state.closed))

    # Stock remaining [66]
    obs = obs.at[66].set(stock_remaining(state) / 10.0)

    # Talon cards remaining [67]
    obs = obs.at[67].set(talon_cards_remaining(state) / 10.0)

    # Points [68-69]
    opponent = 1 - player
    obs = obs.at[68].set(state.points[player] / 120.0)
    obs = obs.at[69].set(state.points[opponent] / 120.0)

    # Tricks won [70-71]
    obs = obs.at[70].set(state.tricks_won[player] / float(TRICKS_PER_GAME))
    obs = obs.at[71].set(state.tricks_won[opponent] / float(TRICKS_PER_GAME))

    # Current player [72]
    obs = obs.at[72].set(jnp.float32(state.current_player == player))

    # Leader [73]
    obs = obs.at[73].set(jnp.float32(state.leader == player))

    # Who closed [74-75]
    obs = obs.at[74].set(jnp.float32(state.closed_by == player))
    obs = obs.at[75].set(jnp.float32(state.closed_by == opponent))

    # Strict rules active [76]
    obs = obs.at[76].set(jnp.float32(strict_rules_active(state)))

    return obs


@jax.jit
def legal_actions_mask(state: SnapszerState) -> jax.Array:
    """Get legal actions as a boolean mask of size [TOTAL_ACTIONS]."""
    mask = jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_)

    # If terminal, no actions
    mask = jnp.where(state.terminal, mask, mask)

    me = state.current_player
    hand_full = state.hands[me]  # Full hand with -1 padding

    # Exchange trump jack
    can_exchange = can_exchange_trump_jack(state, me)
    mask = mask.at[EXCHANGE_TRUMP_ACTION].set(can_exchange)

    # Close talon
    can_close = (
        (~state.closed) &
        (~talon_empty(state)) &
        (stock_remaining(state) >= CLOSE_MIN_ABOVE_TRUMP) &
        (state.leader == me) &
        (state.trick_cards[0] == -1)
    )
    mask = mask.at[CLOSE_TALON_ACTION].set(can_close)

    # Card plays
    is_leader = (state.leader == me) & (state.trick_cards[0] == -1)
    is_replier = state.trick_cards[0] >= 0

    # Leader can play any card in hand (excluding -1 padding)
    def set_leader_actions(mask):
        # Set mask for all valid cards in hand
        valid_cards = hand_full >= 0
        def set_card(i, m):
            return jnp.where(valid_cards[i], m.at[hand_full[i]].set(True), m)
        return jax.lax.fori_loop(0, MAX_HAND_SIZE, set_card, mask)

    # Replier must follow strict rules if active
    def set_replier_actions(mask):
        def strict_reply(mask):
            legal = legal_reply_cards(state.hands[me], state.hand_sizes[me], state.trick_cards[0], state.trump)
            # Set mask for all valid legal cards
            valid_legal = legal >= 0
            def set_card(i, m):
                return jnp.where(valid_legal[i], m.at[legal[i]].set(True), m)
            return jax.lax.fori_loop(0, MAX_HAND_SIZE, set_card, mask)

        def open_reply(mask):
            # Same as leader actions
            valid_cards = hand_full >= 0
            def set_card(i, m):
                return jnp.where(valid_cards[i], m.at[hand_full[i]].set(True), m)
            return jax.lax.fori_loop(0, MAX_HAND_SIZE, set_card, mask)

        return jax.lax.cond(
            strict_rules_active(state),
            strict_reply,
            open_reply,
            mask
        )

    mask = jax.lax.cond(
        is_leader,
        set_leader_actions,
        lambda m: jax.lax.cond(is_replier, set_replier_actions, lambda m: m, m),
        mask
    )

    # If terminal, clear all actions
    mask = jnp.where(state.terminal, jnp.zeros_like(mask), mask)

    return mask


@jax.jit
def legal_actions(state: SnapszerState) -> jax.Array:
    """Get legal actions as a list (padded with -1)."""
    mask = legal_actions_mask(state)
    actions = jnp.where(mask, jnp.arange(TOTAL_ACTIONS), -1)
    # Compact: move all valid actions to the front
    valid = actions >= 0
    return jnp.where(jnp.arange(TOTAL_ACTIONS) < jnp.sum(valid),
                     jnp.sort(jnp.where(valid, actions, TOTAL_ACTIONS))[:TOTAL_ACTIONS],
                     -1)


@jax.jit
def remove_card_from_hand(state: SnapszerState, player: jnp.ndarray, cid: jnp.ndarray) -> SnapszerState:
    """Remove a card from player's hand."""
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]

    # Find first occurrence of target card
    matches = (hand == cid) & (jnp.arange(MAX_HAND_SIZE) < hand_size)
    target_idx = jnp.argmax(matches)  # Index of first match
    has_match = jnp.any(matches)

    # Build new hand: for each position i, determine source
    def get_card(i):
        # If i < target_idx: copy from i
        # If i >= target_idx: copy from i+1
        source_idx = jnp.where(i < target_idx, i, i + 1)
        # Only valid if source_idx < hand_size
        is_valid = source_idx < hand_size
        return jnp.where(is_valid, hand[source_idx], -1)

    new_hand = jax.vmap(get_card)(jnp.arange(MAX_HAND_SIZE))

    # Only update if we found a match
    final_hand = jnp.where(has_match, new_hand, hand)
    final_size = jnp.where(has_match, hand_size - 1, hand_size)
    final_mask = jnp.where(has_match, state.hand_masks[player] & ~(1 << cid), state.hand_masks[player])

    # Update state
    new_hands = state.hands.at[player].set(final_hand)
    new_hand_sizes = state.hand_sizes.at[player].set(final_size)
    new_hand_masks = state.hand_masks.at[player].set(final_mask)

    return state._replace(
        hands=new_hands,
        hand_sizes=new_hand_sizes,
        hand_masks=new_hand_masks
    )


@jax.jit
def insert_card_to_hand(state: SnapszerState, player: jnp.ndarray, cid: jnp.ndarray) -> SnapszerState:
    """Insert a card into player's hand (maintaining sorted order)."""
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]

    # Find insertion position by comparing with all valid cards
    # For each position, check if cid should go there
    def find_insert_pos(i):
        # Count how many valid cards are less than cid
        is_valid = i < hand_size
        card_val = hand[i]
        is_less = card_val < cid
        return is_valid & is_less

    counts = jax.vmap(find_insert_pos)(jnp.arange(MAX_HAND_SIZE))
    insert_pos = jnp.sum(counts)

    # Shift cards right from insert_pos
    def shift_right(i):
        return jnp.where(
            i < insert_pos,
            hand[i],
            jnp.where(i == insert_pos, cid, hand[i - 1])
        )

    new_hand = jax.vmap(shift_right)(jnp.arange(MAX_HAND_SIZE))

    # Update state
    new_hands = state.hands.at[player].set(new_hand)
    new_hand_sizes = state.hand_sizes.at[player].set(hand_size + 1)
    new_mask = state.hand_masks[player] | (1 << cid)
    new_hand_masks = state.hand_masks.at[player].set(new_mask)

    return state._replace(
        hands=new_hands,
        hand_sizes=new_hand_sizes,
        hand_masks=new_hand_masks
    )


@jax.jit
def compute_game_points(state: SnapszerState, winner: jnp.ndarray, cause: jnp.ndarray) -> jax.Array:
    """Compute game points. Cause: 0=closer_fail, 1=durchmarsch, 2=auto66, 3=last_trick."""
    loser = 1 - winner

    # closer_fail: 2 points
    gp_closer_fail = jnp.int32(2)

    # durchmarsch or loser won 0 tricks: 3 points
    gp_durchmarsch = jnp.int32(3)

    # auto66: 3 if loser won 0 tricks, 2 if loser < 33 points, else 1
    gp_auto66 = jnp.where(
        state.tricks_won[loser] == 0,
        jnp.int32(3),
        jnp.where(state.points[loser] < 33, jnp.int32(2), jnp.int32(1))
    )

    # last_trick: 1 if loser >= 33 points, else 2
    gp_last_trick = jnp.where(state.points[loser] >= 33, jnp.int32(1), jnp.int32(2))

    # Select based on cause
    gp = jnp.where(
        cause == 0, gp_closer_fail,
        jnp.where(
            cause == 1, gp_durchmarsch,
            jnp.where(cause == 2, gp_auto66, gp_last_trick)
        )
    )

    result = jnp.array([0, 0], dtype=jnp.int32)
    result = result.at[winner].set(gp)
    return result


@jax.jit
def finalize_game(state: SnapszerState, winner: jnp.ndarray, cause: jnp.ndarray) -> SnapszerState:
    """Finalize the game with a winner. Cause: 0=closer_fail, 1=durchmarsch, 2=auto66, 3=last_trick."""
    # Don't finalize if already terminal
    state = jax.lax.cond(
        state.terminal,
        lambda s: s,
        lambda s: s._replace(
            terminal=jnp.bool_(True),
            winner=winner,
            last_trick_winner=jnp.where(s.last_trick_winner >= 0, s.last_trick_winner, winner),
            current_player=jnp.int32(-1),
            game_points=compute_game_points(s, winner, cause)
        ),
        state
    )
    return state


@jax.jit
def maybe_finalize_on_66(state: SnapszerState) -> SnapszerState:
    """Check if either player reached 66 points and finalize if so."""
    has_winner = (state.points[0] >= 66) | (state.points[1] >= 66)
    winner = jnp.where(state.points[0] >= 66, jnp.int32(0), jnp.int32(1))

    return jax.lax.cond(
        has_winner & ~state.terminal,
        lambda s: finalize_game(s, winner, jnp.int32(2)),  # cause=2 (auto66)
        lambda s: s,
        state
    )


@jax.jit
def score_marriage_if_any(state: SnapszerState, player: jnp.ndarray, played_cid: jnp.ndarray) -> SnapszerState:
    """Score a marriage if conditions are met."""
    # Must not be in strict rules
    # Must be leading the trick (trick_cards[0] is -1)
    # Must be King or Queen
    # Must have the counterpart card

    can_score = (~strict_rules_active(state)) & (state.trick_cards[0] == -1)

    r = cid_rank(played_cid)
    s = cid_suit(played_cid)

    is_king_or_queen = (r == 2) | (r == 3)
    already_scored = state.marriages_scored[player, s]

    # Determine counterpart
    counterpart_rank = jnp.where(r == 2, jnp.int32(3), jnp.int32(2))
    counterpart_cid = card_id(s, counterpart_rank)
    has_counterpart = mask_contains(state.hand_masks[player], counterpart_cid)

    should_score = can_score & is_king_or_queen & ~already_scored & has_counterpart

    # Calculate bonus
    bonus = jnp.where(s == state.trump, 40, 20)

    # Update state
    new_marriages = state.marriages_scored.at[player, s].set(
        state.marriages_scored[player, s] | should_score
    )
    new_points = state.points.at[player].add(jnp.where(should_score, bonus, 0))

    state = state._replace(marriages_scored=new_marriages, points=new_points)

    # Check for 66 after marriage
    state = maybe_finalize_on_66(state)

    return state


@jax.jit
def finish_trick(state: SnapszerState) -> SnapszerState:
    """Finish the current trick and handle card drawing."""
    lead_card = state.trick_cards[0]
    reply_card = state.trick_cards[1]

    # Determine winner
    w_rel = trick_winner(lead_card, reply_card, state.trump)
    w = (state.leader + w_rel) % 2

    # Award points
    pts = card_points_fn(lead_card) + card_points_fn(reply_card)
    new_points = state.points.at[w].add(pts)
    new_tricks_won = state.tricks_won.at[w].add(1)

    state = state._replace(
        points=new_points,
        tricks_won=new_tricks_won,
        last_trick_winner=w,
        trick_cards=jnp.array([-1, -1], dtype=jnp.int32)
    )

    # Check for 66
    state = maybe_finalize_on_66(state)

    # If terminal, return
    state = jax.lax.cond(
        state.terminal,
        lambda s: s,
        lambda s: _handle_card_draw(s, w),
        state
    )

    return state


@jax.jit
def _handle_card_draw(state: SnapszerState, w: jnp.ndarray) -> SnapszerState:
    """Handle card drawing after trick. w is the winner."""
    # Only draw if talon is not closed and not empty
    should_draw = (~state.closed) & (~talon_empty(state))

    def draw_cards(s):
        remaining = stock_remaining(s)

        # Case 1: remaining >= 2, both players draw from stock
        def draw_two(s):
            # Winner draws first
            card1 = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, w, card1)
            s = s._replace(stock_idx=s.stock_idx + 1)

            # Loser draws second
            card2 = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, 1 - w, card2)
            s = s._replace(stock_idx=s.stock_idx + 1)

            return s

        # Case 2: remaining == 1, winner draws last stock card, loser gets trump
        def draw_one(s):
            # Winner draws from stock
            card = s.stock[s.stock_idx]
            s = insert_card_to_hand(s, w, card)
            s = s._replace(stock_idx=s.stock_idx + 1)

            # Loser gets trump card if not taken
            s = jax.lax.cond(
                ~s.trump_taken,
                lambda s: insert_card_to_hand(s, 1 - w, s.trump_card)._replace(trump_taken=jnp.bool_(True)),
                lambda s: s,
                s
            )

            return s

        # Case 3: remaining == 0, only trump card left
        def draw_trump_only(s):
            s = jax.lax.cond(
                ~s.trump_taken,
                lambda s: insert_card_to_hand(s, w, s.trump_card)._replace(trump_taken=jnp.bool_(True)),
                lambda s: s,
                s
            )
            return s

        # Conditional drawing
        s = jax.lax.cond(
            remaining >= 2,
            draw_two,
            lambda s: jax.lax.cond(
                remaining == 1,
                draw_one,
                draw_trump_only,
                s
            ),
            s
        )

        return s

    state = jax.lax.cond(should_draw, draw_cards, lambda s: s, state)

    # Update leader and current player
    state = state._replace(leader=w, current_player=w)

    # Check for durchmarsch (won all tricks)
    state = jax.lax.cond(
        state.tricks_won[w] == TRICKS_PER_GAME,
        lambda s: finalize_game(s, w, jnp.int32(1)),  # cause=1 (durchmarsch)
        lambda s: s,
        state
    )

    # Check if game ends (no cards left)
    both_empty = (state.hand_sizes[0] == 0) & (state.hand_sizes[1] == 0)

    def resolve_last_trick(s):
        # If closer failed to reach 66, opponent wins
        closer_failed = (s.closed_by >= 0) & (s.points[s.closed_by] < 66)
        final_winner = jnp.where(closer_failed, 1 - s.closed_by, w)
        cause = jnp.where(closer_failed, jnp.int32(0), jnp.int32(3))  # 0=closer_fail, 3=last_trick
        return finalize_game(s, final_winner, cause)

    state = jax.lax.cond(
        both_empty & ~state.terminal,
        resolve_last_trick,
        lambda s: s,
        state
    )

    return state


@jax.jit
def exchange_trump_jack(state: SnapszerState, player: jnp.ndarray) -> SnapszerState:
    """Exchange trump jack with face-up trump card."""
    jack_cid = card_id(state.trump, TRUMP_JACK_RANK)

    # Remove jack from hand
    state = remove_card_from_hand(state, player, jack_cid)

    # Add trump card to hand
    gained = state.trump_card
    state = insert_card_to_hand(state, player, gained)

    # Update trump card to jack
    state = state._replace(trump_card=jack_cid)

    return state


@jax.jit
def apply_close(state: SnapszerState) -> SnapszerState:
    """Close the talon."""
    return state._replace(
        closed=jnp.bool_(True),
        closed_by=state.current_player,
        stock_idx=jnp.int32(len(state.stock)),
        trump_taken=jnp.bool_(True)
    )


@jax.jit
def apply_action(state: SnapszerState, action: jnp.ndarray) -> SnapszerState:
    """Apply an action to the state and return new state."""
    # If terminal, do nothing
    state = jax.lax.cond(
        state.terminal,
        lambda s: s,
        lambda s: _apply_action_impl(s, action),
        state
    )
    return state


@jax.jit
def _apply_action_impl(state: SnapszerState, action: jnp.ndarray) -> SnapszerState:
    """Internal implementation of apply_action."""
    me = state.current_player

    # Handle EXCHANGE_TRUMP_ACTION
    def handle_exchange(s):
        return exchange_trump_jack(s, me)

    # Handle CLOSE_TALON_ACTION
    def handle_close(s):
        return apply_close(s)

    # Handle card play
    def handle_card_play(s):
        cid = action

        # Check if leading or replying
        is_leading = (s.leader == me) & (s.trick_cards[0] == -1)

        def lead_card(s):
            # Score marriage if any
            s = score_marriage_if_any(s, me, cid)

            # If game ended due to marriage, return early
            def continue_lead(s):
                # Set trick card
                s = s._replace(trick_cards=s.trick_cards.at[0].set(cid))
                # Remove card from hand (BEFORE switching player!)
                s = remove_card_from_hand(s, me, cid)
                # Switch player
                s = s._replace(current_player=1 - me)
                return s

            s = jax.lax.cond(
                s.terminal,
                lambda s: s,
                continue_lead,
                s
            )

            return s

        def reply_card(s):
            # Set reply card
            s = s._replace(trick_cards=s.trick_cards.at[1].set(cid))

            # Remove card from hand
            s = remove_card_from_hand(s, me, cid)

            # Finish trick
            s = finish_trick(s)

            return s

        return jax.lax.cond(is_leading, lead_card, reply_card, s)

    # Dispatch based on action type
    state = jax.lax.cond(
        action == EXCHANGE_TRUMP_ACTION,
        handle_exchange,
        lambda s: jax.lax.cond(
            action == CLOSE_TALON_ACTION,
            handle_close,
            handle_card_play,
            s
        ),
        state
    )

    return state


def returns(state: SnapszerState) -> Tuple[float, float]:
    """Get returns for both players."""
    if not state.terminal or state.winner == -1:
        return (0.0, 0.0)

    diff = state.game_points[0] - state.game_points[1]
    return (float(diff), float(-diff))
