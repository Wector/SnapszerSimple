# Copied into package: original at repo root; this is the canonical module path
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from bisect import bisect_left, insort
import random

SUITS = [0, 1, 2, 3]
SUIT_NAMES = {0: "S", 1: "H", 2: "D", 3: "C"}
RANKS = [0, 1, 2, 3, 4]  # A,10,K,Q,J
RANK_NAMES = {0: "A", 1: "10", 2: "K", 3: "Q", 4: "J"}
RANK_STRENGTH = {0: 5, 1: 4, 2: 3, 3: 2, 4: 1}
CARD_POINTS = {0: 11, 1: 10, 2: 4, 3: 3, 4: 2}
NUM_RANKS = len(RANKS)
NUM_CARDS = len(SUITS) * NUM_RANKS
TRICKS_PER_GAME = NUM_CARDS // 2
TRUMP_JACK_RANK = NUM_RANKS - 1
EXCHANGE_TRUMP_ACTION = NUM_CARDS
CLOSE_TALON_ACTION = NUM_CARDS + 1
TOTAL_ACTIONS = NUM_CARDS + 2  # cards + EXCHANGE + CLOSE (no CLAIM)
CLOSE_MIN_ABOVE_TRUMP = 3
OBSERVATION_SIZE = 80

def card_id(suit: int, rank: int) -> int:
    return suit * NUM_RANKS + rank

def cid_suit(cid: int) -> int:
    return cid // NUM_RANKS

def cid_rank(cid: int) -> int:
    return cid % NUM_RANKS

def card_points(cid: int) -> int:
    return CARD_POINTS[cid_rank(cid)]

def card_str(cid: int) -> str:
    return f"{RANK_NAMES[cid_rank(cid)]}{SUIT_NAMES[cid_suit(cid)]}"

def cards_to_mask(cards: List[int]) -> int:
    mask = 0
    for cid in cards:
        mask |= 1 << cid
    return mask

def mask_contains(mask: int, cid: int) -> bool:
    return (mask >> cid) & 1 == 1

class _MT19937:
    def __init__(self, seed: int):
        self._mt = [0] * 624
        self._index = 624
        self.seed(seed)

    def seed(self, seed: int):
        seed &= 0xFFFFFFFF
        self._mt[0] = seed
        for i in range(1, 624):
            self._mt[i] = (1812433253 * (self._mt[i - 1] ^ (self._mt[i - 1] >> 30)) + i) & 0xFFFFFFFF

    def _twist(self):
        for i in range(624):
            x = (self._mt[i] & 0x80000000) + (self._mt[(i + 1) % 624] & 0x7FFFFFFF)
            xA = x >> 1
            if x & 1:
                xA ^= 0x9908B0DF
            self._mt[i] = self._mt[(i + 397) % 624] ^ xA
        self._index = 0

    def rand_uint32(self) -> int:
        if self._index >= 624:
            self._twist()
        y = self._mt[self._index]
        self._index += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def random_double(self) -> float:
        a = self.rand_uint32() >> 5
        b = self.rand_uint32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0

def _mt_shuffle(values: List[int], seed: int) -> None:
    rng = _MT19937(seed)
    for i in range(len(values) - 1, 0, -1):
        j = int(rng.random_double() * (i + 1))
        values[i], values[j] = values[j], values[i]

def trick_winner(lead_cid: int, reply_cid: int, trump: int, _post_close_rules: bool) -> int:
    ls, lr = cid_suit(lead_cid), cid_rank(lead_cid)
    rs, rr = cid_suit(reply_cid), cid_rank(reply_cid)
    if rs == ls:
        return 0 if RANK_STRENGTH[lr] >= RANK_STRENGTH[rr] else 1
    if rs == trump and ls != trump:
        return 1
    return 0

def legal_reply_cards(hand: List[int], lead_cid: int, trump: int) -> List[int]:
    lead_s = cid_suit(lead_cid)
    lead_r = cid_rank(lead_cid)
    same_suit: List[int] = []
    beating: List[int] = []
    trumps: List[int] = []
    lead_strength = RANK_STRENGTH[lead_r]
    for cid in hand:
        suit = cid_suit(cid)
        if suit == lead_s:
            same_suit.append(cid)
            if RANK_STRENGTH[cid_rank(cid)] > lead_strength:
                beating.append(cid)
        elif suit == trump:
            trumps.append(cid)
    if same_suit:
        return beating if beating else same_suit
    if trumps:
        return trumps
    return list(hand)

@dataclass
class SnapszerState:
    trump: int
    trump_card: int
    deck: List[int]
    stock: List[int]
    stock_idx: int
    hands: List[List[int]]
    hand_masks: List[int]
    current_player: int
    leader: int
    trick_cards: List[Optional[int]]
    points: List[int]
    tricks_won: List[int]
    closed: bool
    closed_by: Optional[int] = None
    trump_taken: bool = False
    last_trick_winner: Optional[int] = None
    marriages_scored: List[Dict[int, bool]] = field(default_factory=lambda: [{s: False for s in SUITS}, {s: False for s in SUITS}])
    terminal: bool = False
    winner: Optional[int] = None
    game_points: Optional[Tuple[int, int]] = None

    @staticmethod
    def new(seed: Optional[int] = None) -> "SnapszerState":
        deck = list(range(NUM_CARDS))
        if seed is None:
            random.shuffle(deck)
        else:
            _mt_shuffle(deck, int(seed) & 0xFFFFFFFF)
        p0 = sorted(deck[:5])
        p1 = sorted(deck[5:10])
        trump_card = deck[10]
        trump = cid_suit(trump_card)
        stock = deck[11:]
        state = SnapszerState(
            trump=trump,
            trump_card=trump_card,
            deck=deck,
            stock=stock,
            stock_idx=0,
            hands=[p0, p1],
            hand_masks=[cards_to_mask(p0), cards_to_mask(p1)],
            current_player=0,
            leader=0,
            trick_cards=[None, None],
            points=[0, 0],
            tricks_won=[0, 0],
            closed=False,
        )
        return state

    def stock_remaining(self) -> int:
        return len(self.stock) - self.stock_idx

    def talon_cards_remaining(self) -> int:
        return self.stock_remaining() + (0 if self.trump_taken else 1)

    def talon_empty(self) -> bool:
        return self.trump_taken and self.stock_idx >= len(self.stock)

    def strict_rules_active(self) -> bool:
        return self.closed or self.talon_empty()

    def hand_str(self, p: int) -> str:
        return " ".join(card_str(c) for c in sorted(self.hands[p]))

    def public_str(self) -> str:
        stock_size = self.talon_cards_remaining()
        tc0 = card_str(self.trick_cards[0]) if self.trick_cards[0] is not None else "--"
        tc1 = card_str(self.trick_cards[1]) if self.trick_cards[1] is not None else "--"
        up = card_str(self.trump_card) if not self.trump_taken else "--"
        return (
            f"Trump:{SUIT_NAMES[self.trump]} Up:{up} Stock:{stock_size} Closed:{self.closed} "
            f"Leader:P{self.leader} Trick:[{tc0} {tc1}] Pts:{self.points} Tricks:{self.tricks_won} Last:{self.last_trick_winner}"
        )

    def observation_tensor(self, player: int) -> List[float]:
        values = [0.0] * OBSERVATION_SIZE
        for cid in self.hands[player]:
            values[cid] = 1.0
        if self.trick_cards[0] is not None:
            values[20 + self.trick_cards[0]] = 1.0
        if self.trick_cards[1] is not None:
            values[40 + self.trick_cards[1]] = 1.0
        values[60 + self.trump] = 1.0
        values[64] = 1.0 if self.trump_taken else 0.0
        values[65] = 1.0 if self.closed else 0.0
        values[66] = self.stock_remaining() / 10.0
        values[67] = self.talon_cards_remaining() / 10.0
        opponent = 1 - player
        values[68] = self.points[player] / 120.0
        values[69] = self.points[opponent] / 120.0
        values[70] = self.tricks_won[player] / float(TRICKS_PER_GAME)
        values[71] = self.tricks_won[opponent] / float(TRICKS_PER_GAME)
        values[72] = 1.0 if self.current_player == player else 0.0
        values[73] = 1.0 if self.leader == player else 0.0
        values[74] = 1.0 if self.closed_by == player else 0.0
        values[75] = 1.0 if self.closed_by == opponent else 0.0
        values[76] = 1.0 if self.strict_rules_active() else 0.0
        return values

    def to_string(self, perspective: int) -> str:
        return (
            f"You({perspective}) hand: {self.hand_str(perspective)} | Opponent: {len(self.hands[1-perspective])} cards | "
            + self.public_str()
        )

    def is_terminal(self) -> bool:
        return self.terminal

    def returns(self) -> Tuple[float, float]:
        if not self.terminal or self.winner is None:
            return (0.0, 0.0)
        if self.game_points is not None:
            diff = self.game_points[0] - self.game_points[1]
            return (float(diff), float(-diff))
        w = self.winner
        return (1.0 if w == 0 else -1.0, 1.0 if w == 1 else -1.0)

    def legal_actions(self) -> List[int]:
        if self.terminal:
            return []
        actions: List[int] = []
        me = self.current_player
        hand = self.hands[me]
        if self._can_exchange_trump_jack(me):
            actions.append(EXCHANGE_TRUMP_ACTION)
        if (
            not self.closed
            and not self.talon_empty()
            and self.stock_remaining() >= CLOSE_MIN_ABOVE_TRUMP
            and self.leader == me
            and self.trick_cards[0] is None
        ):
            actions.append(CLOSE_TALON_ACTION)
        if self.leader == me and self.trick_cards[0] is None:
            actions.extend(hand)
        else:
            lead_cid = self.trick_cards[0]
            assert lead_cid is not None
            if self.strict_rules_active():
                allowed = legal_reply_cards(hand, lead_cid, self.trump)
                actions.extend(allowed)
            else:
                actions.extend(hand)
        return actions

    def _can_exchange_trump_jack(self, player: int) -> bool:
        if self.closed or self.talon_empty() or self.trump_taken:
            return False
        if self.current_player != player or self.leader != player:
            return False
        if self.trick_cards[0] is not None:
            return False
        if self.stock_remaining() <= 0:
            return False
        jack_cid = card_id(self.trump, TRUMP_JACK_RANK)
        return mask_contains(self.hand_masks[player], jack_cid)

    def _apply_close(self):
        self.closed = True
        self.closed_by = self.current_player
        self.stock_idx = len(self.stock)
        self.trump_taken = True

    def _exchange_trump_jack(self, player: int) -> None:
        jack_cid = card_id(self.trump, TRUMP_JACK_RANK)
        if not mask_contains(self.hand_masks[player], jack_cid):
            raise ValueError("Cannot exchange without trump jack")
        self._remove_card_from_hand(player, jack_cid)
        gained = self.trump_card
        self._insert_card_to_hand(player, gained)
        self.trump_card = jack_cid

    def _maybe_finalize_on_66(self) -> None:
        if self.terminal:
            return
        if self.points[0] >= 66 or self.points[1] >= 66:
            winner = 0 if self.points[0] >= 66 else 1
            self._finalize_game(winner, cause="auto66")

    def _score_marriage_if_any(self, player: int, played_cid: int):
        if self.strict_rules_active():
            return
        if self.trick_cards[0] is not None:
            return
        r = cid_rank(played_cid)
        if r not in (2, 3):
            return
        s = cid_suit(played_cid)
        if self.marriages_scored[player][s]:
            return
        need = 3 if r == 2 else 2
        counterpart_cid = card_id(s, need)
        if mask_contains(self.hand_masks[player], counterpart_cid):
            self.marriages_scored[player][s] = True
            bonus = 40 if s == self.trump else 20
            self.points[player] += bonus
            self._maybe_finalize_on_66()

    def _remove_card_from_hand(self, player: int, cid: int) -> None:
        hand = self.hands[player]
        idx = bisect_left(hand, cid)
        if idx >= len(hand) or hand[idx] != cid:
            raise ValueError("Illegal: card not in hand")
        del hand[idx]
        self.hand_masks[player] &= ~(1 << cid)

    def _insert_card_to_hand(self, player: int, cid: int) -> None:
        insort(self.hands[player], cid)
        self.hand_masks[player] |= 1 << cid

    def _finish_trick(self):
        lead_card, reply_card = self.trick_cards
        assert lead_card is not None and reply_card is not None
        w_rel = trick_winner(lead_card, reply_card, self.trump, self.strict_rules_active())
        w = (self.leader + w_rel) % 2
        pts = card_points(lead_card) + card_points(reply_card)
        self.points[w] += pts
        self.tricks_won[w] += 1
        self.last_trick_winner = w
        self.trick_cards = [None, None]

        self._maybe_finalize_on_66()
        if self.terminal:
            return

        if (not self.closed) and (not self.talon_empty()):
            remaining = self.stock_remaining()
            if remaining >= 2:
                for pl in (w, 1 - w):
                    if self.stock_idx < len(self.stock):
                        cid = self.stock[self.stock_idx]
                        self.stock_idx += 1
                        self._insert_card_to_hand(pl, cid)
            elif remaining == 1:
                cid = self.stock[self.stock_idx]
                self.stock_idx += 1
                self._insert_card_to_hand(w, cid)
                if not self.trump_taken:
                    self._insert_card_to_hand(1 - w, self.trump_card)
                    self.trump_taken = True
            else:
                if not self.trump_taken:
                    self._insert_card_to_hand(w, self.trump_card)
                    self.trump_taken = True

        self.leader = w
        self.current_player = w

        if self.tricks_won[w] == TRICKS_PER_GAME:
            self._finalize_game(w, cause="durchmarsch")
            return

        if len(self.hands[0]) == 0 and len(self.hands[1]) == 0:
            winner, cause = self._resolve_last_trick_winner(w)
            self._finalize_game(winner, cause)

    def _resolve_last_trick_winner(self, default_winner: int) -> Tuple[int, str]:
        if self.closed_by is not None and self.points[self.closed_by] < 66:
            return 1 - self.closed_by, "closer_fail"
        return default_winner, "last_trick"

    def _finalize_game(self, winner: int, cause: str) -> None:
        if self.terminal:
            return
        self.terminal = True
        self.winner = winner
        if self.last_trick_winner is None:
            self.last_trick_winner = winner
        self.current_player = -1
        self.game_points = self._compute_game_points(winner, cause)

    def _compute_game_points(self, winner: int, cause: str) -> Tuple[int, int]:
        loser = 1 - winner
        if cause == "closer_fail":
            gp = 2
        elif cause == "durchmarsch" or self.tricks_won[loser] == 0:
            gp = 3
        elif cause == "auto66":
            gp = 3 if self.tricks_won[loser] == 0 else (2 if self.points[loser] < 33 else 1)
        else:
            gp = 1 if self.points[loser] >= 33 else 2
        return (gp if winner == 0 else 0, gp if winner == 1 else 0)

    def apply_action(self, action: int):
        if self.terminal:
            return
        me = self.current_player
        if action == EXCHANGE_TRUMP_ACTION:
            if not self._can_exchange_trump_jack(me):
                raise ValueError("Illegal trump exchange")
            self._exchange_trump_jack(me)
            return
        if action == CLOSE_TALON_ACTION:
            if (
                self.closed
                or self.talon_empty()
                or self.leader != me
                or self.trick_cards[0] is not None
                or self.stock_remaining() < CLOSE_MIN_ABOVE_TRUMP
            ):
                raise ValueError("Illegal close action")
            self._apply_close()
            return

        cid = action
        if not mask_contains(self.hand_masks[me], cid):
            raise ValueError("Illegal: card not in hand")
        if self.leader != me and self.strict_rules_active():
            lead_cid = self.trick_cards[0]
            assert lead_cid is not None
            allowed = legal_reply_cards(self.hands[me], lead_cid, self.trump)
            if cid not in allowed:
                raise ValueError("Illegal reply under strict rules")
        if self.leader == me and self.trick_cards[0] is None:
            self._score_marriage_if_any(me, cid)
            if self.terminal:
                return
            self.trick_cards[0] = cid
            self._remove_card_from_hand(me, cid)
            self.current_player = 1 - me
        else:
            self.trick_cards[1] = cid
            self._remove_card_from_hand(me, cid)
            self._finish_trick()

    def random_playout(self, seed: Optional[int] = None) -> int:
        rnd = random.Random(seed)
        while not self.is_terminal():
            acts = self.legal_actions()
            cards = [a for a in acts if a < NUM_CARDS]
            if cards:
                a = rnd.choice(cards)
                if EXCHANGE_TRUMP_ACTION in acts and rnd.random() < 0.1:
                    a = EXCHANGE_TRUMP_ACTION
            else:
                a = rnd.choice(acts)
            self.apply_action(a)
        return self.winner if self.winner is not None else -1

if __name__ == "__main__":
    wins = [0, 0]
    n = 100
    for _ in range(n):
        s = SnapszerState.new(seed=None)
        w = s.random_playout(seed=None)
        if w >= 0:
            wins[w] += 1
    print("Random vs Random results in", n, "games:", wins)


