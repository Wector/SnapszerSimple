from snapszer_base import SnapszerState, CLOSE_TALON_ACTION, NUM_CARDS, SUITS, RANKS, card_id, cards_to_mask

def test_closing_rule():
    print("Testing Closing Rule...")
    # Create a state
    s = SnapszerState.new()
    
    # Manually set stock to have 2 cards (plus trump is separate)
    # stock_idx is index into deck. stock is deck[11:].
    # We want stock_remaining() to be 2.
    # stock_remaining = len(stock) - stock_idx
    # So if len(stock) is 9 (20 cards - 10 hands - 1 trump = 9), we need stock_idx = 7.
    s.stock_idx = len(s.stock) - 2
    
    # Ensure leader is current player
    s.leader = s.current_player
    s.trick_cards = [None, None]
    s.closed = False
    s.talon_empty = lambda: False # Mock if needed, but stock_remaining > 0 should suffice for real method
    
    # Check if CLOSE_TALON_ACTION is in legal_actions
    actions = s.legal_actions()
    can_close = CLOSE_TALON_ACTION in actions
    print(f"  Stock remaining: {s.stock_remaining()}")
    print(f"  Can close: {can_close}")
    
    if can_close:
        print("  [PASS] Closing allowed with 2 cards in stock.")
    else:
        print("  [FAIL] Closing NOT allowed with 2 cards in stock (Expected for old rules).")

def test_last_trick_scoring():
    print("\nTesting Last Trick Scoring...")
    s = SnapszerState.new()
    
    # Simulate end of game
    s.terminal = True
    s.winner = 0
    s.last_trick_winner = 0
    s.tricks_won = [3, 2] # Both won tricks
    s.points = [40, 10] # Winner has 40, Loser has 10 (< 33)
    
    # We need to call _compute_game_points directly or via a helper since _finalize_game sets it
    # But _compute_game_points is what we want to test.
    # It takes (winner, cause)
    
    gp_tuple = s._compute_game_points(0, "last_trick")
    gp = gp_tuple[0]
    
    print(f"  Winner Points: {s.points[0]}, Loser Points: {s.points[1]}")
    print(f"  Game Points Awarded: {gp}")
    
    if gp == 1:
        print("  [PASS] Awarded 1 GP for last trick (Loser < 33).")
    elif gp == 2:
        print("  [FAIL] Awarded 2 GP for last trick (Loser < 33) (Expected for old rules).")
    else:
        print(f"  [?] Awarded {gp} GP.")

if __name__ == "__main__":
    try:
        test_closing_rule()
        test_last_trick_scoring()
    except Exception as e:
        print(f"An error occurred: {e}")
