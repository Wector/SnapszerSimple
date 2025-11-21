from snapszer_base import SnapszerState, CLOSE_TALON_ACTION, NUM_CARDS
import random

def measure_talon_availability(num_games=1000):
    total_decisions = 0
    close_available_count = 0
    
    for _ in range(num_games):
        state = SnapszerState.new()
        while not state.is_terminal():
            legal = state.legal_actions()
            
            # Only count decisions where the agent is the leader (active player)
            # because passive players can never close
            if state.leader == state.current_player:
                total_decisions += 1
                if CLOSE_TALON_ACTION in legal:
                    close_available_count += 1
            
            # Play randomly
            action = random.choice(legal)
            state.apply_action(action)
            
    print(f"Games Simulated: {num_games}")
    print(f"Total Leader Decisions: {total_decisions}")
    print(f"Close Talon Available: {close_available_count}")
    print(f"Availability Rate: {close_available_count / total_decisions * 100:.2f}%")

if __name__ == "__main__":
    measure_talon_availability()
