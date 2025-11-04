"""
JAX implementation of Hungarian Snapszer game.
This is a functional, vectorized implementation that matches snapszer_base.py exactly.
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, NamedTuple
import chex

# Game constants (must match snapszer_base.py)
NUM_CARDS = 20
TRICKS_PER_GAME = 10
TRUMP_JACK_RANK = 4
EXCHANGE_TRUMP_ACTION = 20
CLOSE_TALON_ACTION = 21
TOTAL_ACTIONS = 22
CLOSE_MIN_ABOVE_TRUMP = 3
OBSERVATION_SIZE = 80

# Rank strength for winning tricks (A=5, 10=4, K=3, Q=2, J=1)
RANK_STRENGTH = jnp.array([5, 4, 3, 2, 1], dtype=jnp.int32)

# Card points (A=11, 10=10, K=4, Q=3, J=2)
CARD_POINTS = jnp.array([11, 10, 4, 3, 2], dtype=jnp.int32)


# MT19937 Implementation for deterministic shuffling
@jax.jit
def mt19937_init(seed: chex.Array) -> chex.Array:
    """Initialize MT19937 state. Returns array of shape (624,)"""
    mask = jnp.uint32(0xFFFFFFFF)
    seed = jnp.bitwise_and(seed, mask)
    mt = jnp.zeros(624, dtype=jnp.uint32)
    mt = mt.at[0].set(seed)

    def init_step(i, mt):
        prev = mt[i - 1]
        val = jnp.bitwise_and(jnp.uint32(1812433253) * (prev ^ (prev >> 30)) + jnp.uint32(i), mask)
        return mt.at[i].set(val)

    mt = jax.lax.fori_loop(1, 624, init_step, mt)
    return mt


@jax.jit
def mt19937_twist(mt: chex.Array) -> chex.Array:
    """Twist the MT state."""
    def twist_step(i, mt):
        x = jnp.bitwise_and(mt[i], jnp.uint32(0x80000000)) + jnp.bitwise_and(mt[(i + 1) % 624], jnp.uint32(0x7FFFFFFF))
        xA = x >> 1
        xA = jnp.where(jnp.bitwise_and(x, jnp.uint32(1)) != 0, xA ^ jnp.uint32(0x9908B0DF), xA)
        new_val = mt[(i + 397) % 624] ^ xA
        return mt.at[i].set(new_val)

    return jax.lax.fori_loop(0, 624, twist_step, mt)


@jax.jit
def mt19937_extract(mt: chex.Array, index: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Extract a random number. Returns (value, new_mt, new_index)"""
    # Twist if needed
    mt = jax.lax.cond(index >= 624, lambda m: mt19937_twist(m), lambda m: m, mt)
    index = jax.lax.cond(index >= 624, lambda _: jnp.uint32(0), lambda _: index, None)

    y = mt[index]
    index = index + 1

    # Tempering
    y = y ^ (y >> 11)
    y = y ^ (y << 7) & jnp.uint32(0x9D2C5680)
    y = y ^ (y << 15) & jnp.uint32(0xEFC60000)
    y = y ^ (y >> 18)

    return y, mt, index


@jax.jit
def mt19937_random_double(mt: chex.Array, index: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Generate random double in [0, 1). Returns (value, new_mt, new_index)"""
    a, mt, index = mt19937_extract(mt, index)
    b, mt, index = mt19937_extract(mt, index)
    a = a >> 5
    b = b >> 6
    val = (a * 67108864.0 + b) / 9007199254740992.0
    return val, mt, index


@jax.jit
def mt19937_shuffle(seed: chex.Array) -> chex.Array:
    """
    Shuffle cards 0-19 using MT19937, matching base implementation exactly.
    """
    seed_masked = jnp.bitwise_and(seed, jnp.uint32(0xFFFFFFFF))
    mt = mt19937_init(seed_masked)
    index = jnp.uint32(624)  # Force twist on first use

    # Initialize with cards in order
    deck = jnp.arange(NUM_CARDS, dtype=jnp.int32)

    # Fisher-Yates shuffle (backwards) - matches base _mt_shuffle exactly
    def shuffle_step(i, state):
        deck, mt, index = state
        # i ranges from 19 down to 1
        actual_i = 19 - i

        # Generate random double
        rand_val, mt, index = mt19937_random_double(mt, index)

        # j = int(rand_val * (actual_i + 1))
        j = jnp.int32(jnp.floor(rand_val * (actual_i + 1)))
        j = jnp.minimum(j, actual_i)  # Clamp to avoid index issues

        # Swap deck[actual_i] and deck[j]
        temp = deck[actual_i]
        deck = deck.at[actual_i].set(deck[j])
        deck = deck.at[j].set(temp)

        return (deck, mt, index)

    deck, mt, index = jax.lax.fori_loop(0, 19, shuffle_step, (deck, mt, index))
    return deck


class SnapszerState(NamedTuple):
    """JAX-compatible game state using immutable arrays."""
    # Trump and deck
    trump: chex.Array  # int32 scalar: trump suit (0-3)
    trump_card: chex.Array  # int32 scalar: visible trump card ID
    deck: chex.Array  # int32[20]: full shuffled deck

    # Stock/Talon
    stock_idx: chex.Array  # int32 scalar: current position in stock (into deck[11:])

    # Player hands (bitmask representation)
    hand_masks: chex.Array  # int32[2]: bitmasks for player hands

    # Game flow
    current_player: chex.Array  # int32 scalar: whose turn (0 or 1)
    leader: chex.Array  # int32 scalar: who leads the trick
    trick_cards: chex.Array  # int32[2]: [lead_card, reply_card] (-1 if not played)

    # Scoring
    points: chex.Array  # int32[2]: points for each player
    tricks_won: chex.Array  # int32[2]: tricks won by each player

    # Game state flags
    closed: chex.Array  # bool: is stock closed?
    closed_by: chex.Array  # int32: who closed (-1 if not closed)
    trump_taken: chex.Array  # bool: has trump been taken from talon?
    last_trick_winner: chex.Array  # int32: who won last trick (-1 if none)
    marriages_scored: chex.Array  # bool[2, 4]: marriage bonuses per player/suit

    # Terminal state
    terminal: chex.Array  # bool: is game over?
    winner: chex.Array  # int32: winning player (-1 if not terminal)
    game_points: chex.Array  # int32[2]: game points awarded


# Card utility functions
@jax.jit
def card_id(suit: int, rank: int) -> int:
    """Convert suit and rank to card ID (0-19)."""
    return suit * 5 + rank


@jax.jit
def cid_suit(cid: chex.Array) -> chex.Array:
    """Extract suit from card ID."""
    return cid // 5


@jax.jit
def cid_rank(cid: chex.Array) -> chex.Array:
    """Extract rank from card ID."""
    return cid % 5


@jax.jit
def get_card_points(cid: chex.Array) -> chex.Array:
    """Get point value of a card."""
    rank = cid_rank(cid)
    return CARD_POINTS[rank]


@jax.jit
def mask_contains(mask: chex.Array, cid: chex.Array) -> chex.Array:
    """Check if bitmask contains card."""
    return (mask >> cid) & 1 == 1


@jax.jit
def mask_add(mask: chex.Array, cid: chex.Array) -> chex.Array:
    """Add card to bitmask."""
    return mask | (1 << cid)


@jax.jit
def mask_remove(mask: chex.Array, cid: chex.Array) -> chex.Array:
    """Remove card from bitmask."""
    return mask & ~(1 << cid)


@jax.jit
def mask_count(mask: chex.Array) -> chex.Array:
    """Count number of cards in bitmask using Brian Kernighan's algorithm."""
    # Count bits manually since jnp.popcount doesn't exist in older JAX versions
    count = jnp.int32(0)
    m = jnp.uint32(mask)

    # Use a loop to count set bits
    def count_step(i, state):
        count, m = state
        # Check if bit i is set
        bit_set = (m >> i) & jnp.uint32(1)
        return (count + jnp.int32(bit_set), m)

    count, _ = jax.lax.fori_loop(0, 20, count_step, (count, m))
    return count


# Game logic functions
@jax.jit
def trick_winner(lead_cid: chex.Array, reply_cid: chex.Array,
                 trump: chex.Array, post_close_rules: chex.Array) -> chex.Array:
    """
    Determine winner of a trick.
    Returns 0 if leader wins, 1 if follower wins.
    Matches snapszer_base.py:91-98 exactly.
    """
    lead_suit = cid_suit(lead_cid)
    reply_suit = cid_suit(reply_cid)
    lead_rank = cid_rank(lead_cid)
    reply_rank = cid_rank(reply_cid)

    # If same suit: compare strength (leader wins ties with >=)
    same_suit_winner = jnp.where(
        RANK_STRENGTH[lead_rank] >= RANK_STRENGTH[reply_rank],
        0,  # leader wins
        1   # follower wins
    )

    # If different suits:
    # Reply wins if reply is trump and lead is not trump
    # Otherwise leader wins
    diff_suit_winner = jnp.where(
        (reply_suit == trump) & (lead_suit != trump),
        1,  # follower's trump beats
        0   # leader wins
    )

    return jnp.where(reply_suit == lead_suit, same_suit_winner, diff_suit_winner)


@jax.jit
def legal_reply_cards_mask(hand_mask: chex.Array, lead_cid: chex.Array,
                           trump: chex.Array) -> chex.Array:
    """
    Get bitmask of legal reply cards under STRICT rules.
    This matches snapszer_base.py:100-119 exactly.
    """
    lead_suit = cid_suit(lead_cid)
    lead_rank = cid_rank(lead_cid)
    lead_strength = RANK_STRENGTH[lead_rank]

    # Scan all cards
    all_cards = jnp.arange(NUM_CARDS, dtype=jnp.int32)
    in_hand = (hand_mask >> all_cards) & 1 == 1

    suits = cid_suit(all_cards)
    ranks = cid_rank(all_cards)
    strengths = RANK_STRENGTH[ranks]

    # Same suit cards
    same_suit = (suits == lead_suit) & in_hand
    same_suit_mask = jnp.sum(jnp.where(same_suit, 1 << all_cards, 0))

    # Beating cards (same suit, higher strength)
    can_beat = same_suit & (strengths > lead_strength)
    can_beat_mask = jnp.sum(jnp.where(can_beat, 1 << all_cards, 0))

    # Trump cards
    is_trump = (suits == trump) & in_hand
    trump_mask = jnp.sum(jnp.where(is_trump, 1 << all_cards, 0))

    # Logic from base:
    # if same_suit: return beating if beating else same_suit
    # if trumps: return trumps
    # return hand
    legal = jnp.where(
        same_suit_mask != 0,
        jnp.where(can_beat_mask != 0, can_beat_mask, same_suit_mask),
        jnp.where(trump_mask != 0, trump_mask, hand_mask)
    )

    return legal


@jax.jit
def get_stock_remaining(state: SnapszerState) -> chex.Array:
    """Get number of cards remaining in stock (deck[11:] - already drawn)."""
    # stock has 9 cards total (deck[11:20])
    # stock_idx starts at 0 and increments
    return jnp.maximum(9 - state.stock_idx, 0)


@jax.jit
def get_talon_cards_remaining(state: SnapszerState) -> chex.Array:
    """Get total cards in talon (stock + trump card if not taken)."""
    stock_remaining = get_stock_remaining(state)
    trump_available = ~state.trump_taken
    return stock_remaining + jnp.where(trump_available, 1, 0)


@jax.jit
def talon_empty(state: SnapszerState) -> chex.Array:
    """Check if talon is empty (trump taken and no stock left)."""
    return state.trump_taken & (state.stock_idx >= 9)


@jax.jit
def strict_rules_active(state: SnapszerState) -> chex.Array:
    """Check if strict following rules are active (closed or talon empty)."""
    return state.closed | talon_empty(state)


@jax.jit
def can_exchange_trump_jack(state: SnapszerState, player: chex.Array) -> chex.Array:
    """Check if player can exchange trump jack. Matches base:269-279."""
    # Must not be closed, talon must not be empty, trump not taken
    basic_check = ~state.closed & ~talon_empty(state) & ~state.trump_taken

    # Must be current player and leader
    is_current_leader = (state.current_player == player) & (state.leader == player)

    # Must be at start of trick
    at_trick_start = state.trick_cards[0] == -1

    # Stock must have cards remaining
    has_stock = get_stock_remaining(state) > 0

    # Must have trump jack in hand
    trump_jack_id = state.trump * 5 + TRUMP_JACK_RANK
    has_jack = mask_contains(state.hand_masks[player], trump_jack_id)

    return basic_check & is_current_leader & at_trick_start & has_stock & has_jack


@jax.jit
def can_close_talon(state: SnapszerState, player: chex.Array) -> chex.Array:
    """Check if player can close the talon. Matches base:249-256."""
    # Must not be closed, talon not empty
    not_closed = ~state.closed & ~talon_empty(state)

    # Must have enough stock
    enough_stock = get_stock_remaining(state) >= CLOSE_MIN_ABOVE_TRUMP

    # Must be leader
    is_leader = state.leader == player

    # Must be at start of trick
    at_trick_start = state.trick_cards[0] == -1

    return not_closed & enough_stock & is_leader & at_trick_start


@jax.jit
def get_legal_actions(state: SnapszerState) -> chex.Array:
    """
    Get bitmask of legal actions.
    Actions 0-19: play card
    Action 20: exchange trump jack
    Action 21: close talon
    Matches base:241-267.
    """
    player = state.current_player
    hand_mask = state.hand_masks[player]

    # Check if we're leading or following
    is_leading = state.trick_cards[0] == -1

    # Legal card plays
    def leading_cards():
        return hand_mask

    def following_cards():
        strict = strict_rules_active(state)
        return jax.lax.cond(
            strict,
            lambda: legal_reply_cards_mask(hand_mask, state.trick_cards[0], state.trump),
            lambda: hand_mask
        )

    legal_cards = jax.lax.cond(is_leading, leading_cards, following_cards)

    # Special actions
    can_exchange = can_exchange_trump_jack(state, player)
    can_close = can_close_talon(state, player)

    # Combine into single bitmask
    special_actions = (
        jnp.where(can_exchange, 1 << EXCHANGE_TRUMP_ACTION, 0) |
        jnp.where(can_close, 1 << CLOSE_TALON_ACTION, 0)
    )

    return legal_cards | special_actions


@jax.jit
def check_marriage(state: SnapszerState, player: chex.Array,
                   played_cid: chex.Array) -> Tuple[chex.Array, chex.Array]:
    """
    Check if playing this card scores a marriage.
    Returns (has_marriage, marriage_points).
    Matches base:303-320.
    """
    # Can only score if strict rules not active
    can_score = ~strict_rules_active(state)

    # Must be leading (trick_cards[0] is None)
    is_leading = state.trick_cards[0] == -1
    can_score = can_score & is_leading

    played_suit = cid_suit(played_cid)
    played_rank = cid_rank(played_cid)

    # Marriage requires K(2) or Q(3)
    is_k_or_q = (played_rank == 2) | (played_rank == 3)

    # Find partner: if K(2) then need Q(3), if Q(3) then need K(2)
    partner_rank = jnp.where(played_rank == 2, 3, 2)
    partner_cid = played_suit * 5 + partner_rank

    # Check if partner is in hand
    has_partner = mask_contains(state.hand_masks[player], partner_cid)

    # Check if already scored for this suit
    already_scored = state.marriages_scored[player, played_suit]

    # Marriage is valid if all conditions met
    valid_marriage = can_score & is_k_or_q & has_partner & ~already_scored

    # Calculate points (40 for trump, 20 for others)
    is_trump_suit = played_suit == state.trump
    marriage_points = jnp.where(is_trump_suit, 40, 20)

    return valid_marriage, jnp.where(valid_marriage, marriage_points, 0)


@jax.jit
def init_game(rng: chex.PRNGKey) -> SnapszerState:
    """Initialize a new game with given random key."""
    # Convert JAX PRNGKey to seed for MT19937
    # Use the second element of the key as seed (PRNGKey(n) = [0, n])
    seed = jnp.uint32(rng[1])

    # Shuffle deck using MT19937 to match base implementation
    deck = mt19937_shuffle(seed)

    # Cards 0-4: player 0
    # Cards 5-9: player 1
    # Card 10: trump card
    # Cards 11-19: stock

    trump_card = deck[10]
    trump = cid_suit(trump_card)

    # Initialize hand masks (hands are sorted in base, but we only use masks)
    hand0 = deck[0:5]
    hand1 = deck[5:10]

    hand_mask0 = jnp.sum(1 << hand0)
    hand_mask1 = jnp.sum(1 << hand1)
    hand_masks = jnp.array([hand_mask0, hand_mask1], dtype=jnp.int32)

    # Starting player is 0 (non-dealer)
    starting_player = jnp.int32(0)

    return SnapszerState(
        trump=trump,
        trump_card=trump_card,
        deck=deck,
        stock_idx=jnp.int32(0),  # Into deck[11:], so first draw is deck[11]
        hand_masks=hand_masks,
        current_player=starting_player,
        leader=starting_player,
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
def step(state: SnapszerState, action: chex.Array) -> SnapszerState:
    """
    Apply an action to the game state.
    Returns new state.
    """
    # Don't modify terminal states
    def terminal_state(s):
        return s

    def non_terminal_state(s):
        # Handle special actions
        def handle_exchange(s):
            player = s.current_player
            trump_jack_id = s.trump * 5 + TRUMP_JACK_RANK

            # Remove jack from hand, add trump card
            new_hand_mask = mask_remove(s.hand_masks[player], trump_jack_id)
            new_hand_mask = mask_add(new_hand_mask, s.trump_card)

            new_hand_masks = s.hand_masks.at[player].set(new_hand_mask)

            return s._replace(
                hand_masks=new_hand_masks,
                trump_card=trump_jack_id,
            )

        def handle_close(s):
            # Matches base:281-285
            return s._replace(
                closed=jnp.bool_(True),
                closed_by=s.current_player,
                stock_idx=jnp.int32(9),  # Exhaust stock
                trump_taken=jnp.bool_(True),  # Trump is now unavailable
            )

        def handle_card_play(s):
            return play_card(s, action)

        # Route to appropriate handler
        s = jax.lax.cond(
            action == EXCHANGE_TRUMP_ACTION,
            handle_exchange,
            lambda s: jax.lax.cond(
                action == CLOSE_TALON_ACTION,
                handle_close,
                handle_card_play,
                s
            ),
            s
        )

        return s

    return jax.lax.cond(state.terminal, terminal_state, non_terminal_state, state)


@jax.jit
def play_card(state: SnapszerState, card_id: chex.Array) -> SnapszerState:
    """Play a card and update game state."""
    player = state.current_player

    # Remove card from hand
    new_hand_mask = mask_remove(state.hand_masks[player], card_id)
    new_hand_masks = state.hand_masks.at[player].set(new_hand_mask)

    # Check if leading or following
    is_leading = state.trick_cards[0] == -1

    def lead_card():
        """Handle leading a card. Matches base:437-443."""
        # Check for marriage
        has_marriage, marriage_pts = check_marriage(state, player, card_id)

        new_points = state.points.at[player].add(marriage_pts)
        new_marriages = jax.lax.cond(
            has_marriage,
            lambda: state.marriages_scored.at[player, cid_suit(card_id)].set(True),
            lambda: state.marriages_scored
        )

        # Check if player won by reaching 66 (base:296-301)
        won_by_66 = new_points[player] >= 66

        def finalize_66():
            # Winner gets game points based on base:395-405
            opponent = 1 - player
            opp_pts = new_points[opponent]
            opp_tricks = state.tricks_won[opponent]

            # "auto66" cause: gp = 3 if loser has 0 tricks, 2 if <33 points, else 1
            gp = jnp.where(
                opp_tricks == 0,
                3,
                jnp.where(opp_pts < 33, 2, 1)
            )

            game_pts = jnp.zeros(2, dtype=jnp.int32)
            game_pts = game_pts.at[player].set(gp)

            return state._replace(
                hand_masks=new_hand_masks,
                trick_cards=state.trick_cards.at[0].set(card_id),
                points=new_points,
                marriages_scored=new_marriages,
                terminal=jnp.bool_(True),
                winner=player,
                game_points=game_pts,
                last_trick_winner=player,
                current_player=jnp.int32(-1),  # Base sets to -1 on terminal
            )

        def continue_game():
            return state._replace(
                hand_masks=new_hand_masks,
                trick_cards=state.trick_cards.at[0].set(card_id),
                current_player=jnp.int32(1 - player),
                points=new_points,
                marriages_scored=new_marriages,
            )

        return jax.lax.cond(won_by_66, finalize_66, continue_game)

    def follow_card():
        """Handle following to a lead card. Matches base:444-447."""
        lead_card = state.trick_cards[0]

        # Determine trick winner
        winner_idx = trick_winner(lead_card, card_id, state.trump, strict_rules_active(state))
        # winner_idx is 0 (leader) or 1 (follower=current player)
        trick_winner_player = jnp.where(winner_idx == 0, state.leader, player)

        # Calculate trick points
        lead_pts = get_card_points(lead_card)
        follow_pts = get_card_points(card_id)
        trick_pts = lead_pts + follow_pts

        # Award points to winner
        new_points = state.points.at[trick_winner_player].add(trick_pts)
        new_tricks_won = state.tricks_won.at[trick_winner_player].add(1)

        # Update state after trick
        state_after_trick = state._replace(
            hand_masks=new_hand_masks,
            trick_cards=state.trick_cards.at[1].set(card_id),
            points=new_points,
            tricks_won=new_tricks_won,
            last_trick_winner=trick_winner_player,
        )

        # Finish the trick (draw cards, check game end, etc.)
        return finish_trick(state_after_trick)

    return jax.lax.cond(is_leading, lead_card, follow_card)


@jax.jit
def finish_trick(state: SnapszerState) -> SnapszerState:
    """Complete a trick: check for 66, draw cards, check game end. Matches base:334-378."""
    winner = state.last_trick_winner
    loser = 1 - winner

    # Check if winner reached 66 (base:345)
    winner_reached_66 = state.points[winner] >= 66

    # Check if winner won all tricks (durchmarsch)
    durchmarsch = state.tricks_won[winner] == TRICKS_PER_GAME

    # Check if hands are empty
    hands_empty = (mask_count(state.hand_masks[0]) == 0) & (mask_count(state.hand_masks[1]) == 0)

    # Determine if game should end
    # From base:345-378
    should_end = winner_reached_66 | durchmarsch | hands_empty

    def end_game():
        """Finalize the game. Matches base:385-405."""
        # First determine actual winner (may differ from trick winner due to closer_fail)
        # From base:380-383
        closer_failed = (state.closed_by != -1) & (state.points[state.closed_by] < 66)

        # If closer failed, opponent wins (1 - closer)
        actual_winner = jnp.where(
            closer_failed,
            1 - state.closed_by,
            winner  # Otherwise, trick winner is game winner
        )

        actual_loser = 1 - actual_winner
        loser_pts = state.points[actual_loser]
        loser_tricks = state.tricks_won[actual_loser]

        # Determine game points
        # From base:395-405
        # closer_fail: 2
        # durchmarsch or loser has 0 tricks: 3
        # auto66: 3 if loser has 0 tricks, 2 if <33 pts, else 1
        # last_trick: 2 if loser <33, else 1
        game_pts_value = jnp.where(
            closer_failed,
            2,  # Closer failed: 2 points to opponent
            jnp.where(
                durchmarsch | (loser_tricks == 0),
                3,
                jnp.where(
                    winner_reached_66,
                    jnp.where(loser_tricks == 0, 3, jnp.where(loser_pts < 33, 2, 1)),
                    jnp.where(loser_pts < 33, 2, 1)
                )
            )
        )

        game_pts = jnp.zeros(2, dtype=jnp.int32)
        game_pts = game_pts.at[actual_winner].set(game_pts_value)

        return state._replace(
            terminal=jnp.bool_(True),
            winner=actual_winner,
            game_points=game_pts,
            trick_cards=jnp.array([-1, -1], dtype=jnp.int32),
            leader=actual_winner,
            current_player=jnp.int32(-1),  # Base sets to -1 on terminal
        )

    def continue_game():
        """Draw cards and continue. Matches base:349-370."""
        # Draw cards from stock if not closed and not empty
        can_draw = ~state.closed & ~talon_empty(state)

        def draw_cards():
            """Winner draws first, then loser. Matches base:350-367."""
            stock_remaining = get_stock_remaining(state)

            def draw_two():
                # Both draw from stock
                winner_card = state.deck[11 + state.stock_idx]
                loser_card = state.deck[11 + state.stock_idx + 1]

                new_winner_mask = mask_add(state.hand_masks[winner], winner_card)
                new_loser_mask = mask_add(state.hand_masks[loser], loser_card)

                new_hand_masks = jnp.array([
                    jnp.where(winner == 0, new_winner_mask, new_loser_mask),
                    jnp.where(winner == 1, new_winner_mask, new_loser_mask),
                ], dtype=jnp.int32)

                return new_hand_masks, state.stock_idx + 2, state.trump_taken

            def draw_one_and_trump():
                # Winner draws from stock, loser gets trump
                winner_card = state.deck[11 + state.stock_idx]

                new_winner_mask = mask_add(state.hand_masks[winner], winner_card)
                new_loser_mask = mask_add(state.hand_masks[loser], state.trump_card)

                new_hand_masks = jnp.array([
                    jnp.where(winner == 0, new_winner_mask, new_loser_mask),
                    jnp.where(winner == 1, new_winner_mask, new_loser_mask),
                ], dtype=jnp.int32)

                return new_hand_masks, state.stock_idx + 1, jnp.bool_(True)

            def draw_trump_only():
                # Only trump left, winner gets it
                new_winner_mask = mask_add(state.hand_masks[winner], state.trump_card)

                new_hand_masks = state.hand_masks.at[winner].set(new_winner_mask)

                return new_hand_masks, state.stock_idx, jnp.bool_(True)

            # Choose based on stock_remaining
            new_hand_masks, new_stock_idx, new_trump_taken = jax.lax.cond(
                stock_remaining >= 2,
                draw_two,
                lambda: jax.lax.cond(
                    stock_remaining == 1,
                    draw_one_and_trump,
                    draw_trump_only
                )
            )

            return new_hand_masks, new_stock_idx, new_trump_taken

        def no_draw():
            return state.hand_masks, state.stock_idx, state.trump_taken

        new_hand_masks, new_stock_idx, new_trump_taken = jax.lax.cond(
            can_draw,
            draw_cards,
            no_draw
        )

        return state._replace(
            hand_masks=new_hand_masks,
            stock_idx=new_stock_idx,
            trump_taken=new_trump_taken,
            trick_cards=jnp.array([-1, -1], dtype=jnp.int32),
            leader=winner,
            current_player=winner,
        )

    return jax.lax.cond(should_end, end_game, continue_game)


@jax.jit
def get_observation(state: SnapszerState, player: chex.Array) -> chex.Array:
    """
    Get observation tensor for a player.
    Must match snapszer_base.py:198-221 exactly (80 features).
    """
    opponent = 1 - player

    # Features 0-19: player's hand (one-hot for each card)
    hand_bits = jnp.array([
        jnp.float32((state.hand_masks[player] >> i) & 1)
        for i in range(NUM_CARDS)
    ])

    # Features 20-39: trick card 0 (one-hot)
    trick0_bits = jnp.array([
        jnp.float32((state.trick_cards[0] >= 0) & (state.trick_cards[0] == i))
        for i in range(NUM_CARDS)
    ])

    # Features 40-59: trick card 1 (one-hot)
    trick1_bits = jnp.array([
        jnp.float32((state.trick_cards[1] >= 0) & (state.trick_cards[1] == i))
        for i in range(NUM_CARDS)
    ])

    # Features 60-63: trump suit (one-hot)
    trump_bits = jnp.array([
        jnp.float32(state.trump == i)
        for i in range(4)
    ])

    # Feature 64: trump taken
    trump_taken_bit = jnp.float32(state.trump_taken)

    # Feature 65: closed
    closed_bit = jnp.float32(state.closed)

    # Feature 66: stock remaining (normalized)
    stock_remaining_norm = jnp.float32(get_stock_remaining(state)) / 10.0

    # Feature 67: talon cards remaining (normalized)
    talon_remaining_norm = jnp.float32(get_talon_cards_remaining(state)) / 10.0

    # Feature 68: player's points (normalized)
    player_points_norm = jnp.float32(state.points[player]) / 120.0

    # Feature 69: opponent's points (normalized)
    opponent_points_norm = jnp.float32(state.points[opponent]) / 120.0

    # Feature 70: player's tricks won (normalized)
    player_tricks_norm = jnp.float32(state.tricks_won[player]) / jnp.float32(TRICKS_PER_GAME)

    # Feature 71: opponent's tricks won (normalized)
    opponent_tricks_norm = jnp.float32(state.tricks_won[opponent]) / jnp.float32(TRICKS_PER_GAME)

    # Feature 72: is current player
    is_current_bit = jnp.float32(state.current_player == player)

    # Feature 73: is leader
    is_leader_bit = jnp.float32(state.leader == player)

    # Feature 74: closed by player
    closed_by_player_bit = jnp.float32(state.closed_by == player)

    # Feature 75: closed by opponent
    closed_by_opponent_bit = jnp.float32(state.closed_by == opponent)

    # Feature 76: strict rules active
    strict_rules_bit = jnp.float32(strict_rules_active(state))

    # Features 77-79: unused (padding)
    unused_bits = jnp.zeros(3, dtype=jnp.float32)

    # Concatenate all features (total 80)
    return jnp.concatenate([
        hand_bits,          # 0-19
        trick0_bits,        # 20-39
        trick1_bits,        # 40-59
        trump_bits,         # 60-63
        jnp.array([trump_taken_bit]),  # 64
        jnp.array([closed_bit]),        # 65
        jnp.array([stock_remaining_norm]),  # 66
        jnp.array([talon_remaining_norm]),  # 67
        jnp.array([player_points_norm]),    # 68
        jnp.array([opponent_points_norm]),  # 69
        jnp.array([player_tricks_norm]),    # 70
        jnp.array([opponent_tricks_norm]),  # 71
        jnp.array([is_current_bit]),        # 72
        jnp.array([is_leader_bit]),         # 73
        jnp.array([closed_by_player_bit]),  # 74
        jnp.array([closed_by_opponent_bit]), # 75
        jnp.array([strict_rules_bit]),      # 76
        unused_bits,        # 77-79
    ])


# Vectorized batch functions
@jax.jit
def batch_init_games(rng: chex.PRNGKey, batch_size: int) -> SnapszerState:
    """Initialize a batch of games."""
    rngs = random.split(rng, batch_size)
    return jax.vmap(init_game)(rngs)


@jax.jit
def batch_step(states: SnapszerState, actions: chex.Array) -> SnapszerState:
    """Apply actions to a batch of game states."""
    return jax.vmap(step)(states, actions)


@jax.jit
def batch_get_observations(states: SnapszerState, players: chex.Array) -> chex.Array:
    """Get observations for a batch of games."""
    return jax.vmap(get_observation)(states, players)


@jax.jit
def batch_get_legal_actions(states: SnapszerState) -> chex.Array:
    """Get legal action masks for a batch of games."""
    return jax.vmap(get_legal_actions)(states)
