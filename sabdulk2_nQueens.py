#  I would predict that this program could consistently solve an n-queens problem with n = 40
#  in one minute or less on my hardware(my laptop is pretty slow, esp with 10000 lops).

import random
import sys
import math
import time

def initialize_board(n):
    return [random.randint(0, n-1) for _ in range(n)]

def calculate_conflicts(board):
    n = len(board)
    conflicts = 0
    for i in range(n):
        for j in range(i+1, n):
            # Check for same row or diagonal conflicts
            if board[i] == board[j] or abs(board[i] - board[j]) == j - i:
                conflicts += 1
    return conflicts

def get_best_neighbor(board):
    n = len(board)
    best_board = board[:]
    min_conflicts = calculate_conflicts(board)

    for col in range(n):
        original_row = board[col]
        for row in range(n):
            if row != original_row:
                board[col] = row
                conflicts = calculate_conflicts(board)
                if conflicts < min_conflicts:
                    best_board = board[:]
                    min_conflicts = conflicts
        board[col] = original_row

    return best_board, min_conflicts

def get_random_neighbor(board):
    n = len(board)
    col = random.randint(0, n-1)
    new_row = random.randint(0, n-1)
    while new_row == board[col]:
        new_row = random.randint(0, n-1)
    board[col] = new_row
    return board

def hill_climbing(board, max_iterations=1000):
    """Perform hill climbing search to find a solution."""
    n = len(board)  # Get the size of the board
    max_iterations = max(n * 100, max_iterations)  # Ensure enough iterations for small boards
    current_board = board  # Start with the given board
    current_conflicts = calculate_conflicts(current_board)  # Calculate initial conflicts

    for _ in range(max_iterations):  # Loop for the specified number of iterations
        if current_conflicts == 0:  # If a solution is found
            return current_board, current_conflicts  # Return the solution

        new_board, new_conflicts = get_best_neighbor(current_board)  # Get the best neighboring board
        if new_conflicts >= current_conflicts:  # If no improvement is found
            return current_board, current_conflicts  # Return the current best board

        current_board = new_board  # Update the current board
        current_conflicts = new_conflicts  # Update the current conflict count

    return current_board, current_conflicts  # Return the best board found after all iterations

def simulated_annealing(board, max_iterations=10000, initial_temperature=100, cooling_rate=0.995):
    """Perform simulated annealing search to find a solution."""
    n = len(board)  # Get the size of the board
    max_iterations = max(n * 1000, max_iterations)  # Ensure enough iterations for small boards
    current_board = board  # Start with the given board
    current_conflicts = calculate_conflicts(current_board)  # Calculate initial conflicts
    temperature = initial_temperature  # Set the initial temperature

    for _ in range(max_iterations):  # Loop for the specified number of iterations
        if current_conflicts == 0:  # If a solution is found
            return current_board, current_conflicts  # Return the solution

        temperature *= cooling_rate  # Decrease the temperature
        new_board = get_random_neighbor(current_board[:])  # Get a random neighboring board
        new_conflicts = calculate_conflicts(new_board)  # Calculate conflicts for the new board

        # Decide whether to accept the new board
        if new_conflicts < current_conflicts or random.random() < math.exp((current_conflicts - new_conflicts) / temperature):
            current_board = new_board  # Accept the new board
            current_conflicts = new_conflicts  # Update the current conflict count

    return current_board, current_conflicts  # Return the best board found after all iterations

def genetic_algorithm(board, population_size=100, generations=100, mutation_rate=0.1):
    """Perform genetic algorithm search to find a solution."""
    n = len(board)  # Get the size of the board

    def crossover(parent1, parent2):
        """Perform crossover between two parent boards."""
        crossover_point = random.randint(1, n - 1)  # Choose a random crossover point
        return parent1[:crossover_point] + parent2[crossover_point:]  # Combine parents

    def mutate(board):
        """Potentially mutate a board by randomly changing one queen's position."""
        if random.random() < mutation_rate:  # Decide whether to mutate
            col = random.randint(0, n - 1)  # Choose a random column
            board[col] = random.randint(0, n - 1)  # Assign a new random row in that column
        return board

    # Initialize population with random boards and the given board
    population = [initialize_board(n) for _ in range(population_size - 1)] + [board]

    for _ in range(generations):  # Loop for the specified number of generations
        population = sorted(population, key=calculate_conflicts)  # Sort population by fitness
        if calculate_conflicts(population[0]) == 0:  # If a solution is found
            return population[0], 0  # Return the solution

        new_population = population[:2]  # Keep the two best boards (elitism)
        while len(new_population) < population_size:  # Generate new population
            parent1, parent2 = random.choices(population[:50], k=2)  # Select two parents
            child = crossover(parent1, parent2)  # Create a child through crossover
            child = mutate(child)  # Potentially mutate the child
            new_population.append(child)  # Add the child to the new population

        population = new_population  # Replace the old population with the new one

    best_board = min(population, key=calculate_conflicts)  # Find the best board in the final population
    return best_board, calculate_conflicts(best_board)  # Return the best board and its conflict count

def local_beam_search(board, beam_width=50, max_iterations=1000):
    """Perform local beam search to find a solution."""
    n = len(board)  # Get the size of the board
    # Initialize beam with random boards and the given board
    beam = [initialize_board(n) for _ in range(beam_width - 1)] + [board]

    for _ in range(max_iterations):  # Loop for the specified number of iterations
        candidates = []  # Initialize list for candidate boards
        for current_board in beam:  # For each board in the beam
            if calculate_conflicts(current_board) == 0:  # If a solution is found
                return current_board, 0  # Return the solution
            # Generate two random neighbors for each board in the beam
            candidates.extend([get_random_neighbor(current_board[:]) for _ in range(2)])

        # Select the best boards from the candidates to form the new beam
        beam = sorted(candidates, key=calculate_conflicts)[:beam_width]

    best_board = min(beam, key=calculate_conflicts)  # Find the best board in the final beam
    return best_board, calculate_conflicts(best_board)  # Return the best board and its conflict count

def local_search(n, max_iterations=10000):
    """
        Your code goes here
        A couple things to note:
            1. Be mindful of the max_iterations variable. I'm leaving it set in
               here, but you may need to remove it to solve large boards
            2. There's no requirement to print the board every iteration, but
               I think it can help debug.
            3. The only requirement for your local_search function is that it
               needs to take (only) a board size, n, and to return a solved
               board
            4. Feel free to use any of the helper functions I have here, but
               you may need to write extra functions to help. My only
               requirement is that everything stays in this single file.
    """
    """Perform adaptive local search using multiple strategies.

    This function implements an adaptive approach that switches between different
    search strategies based on their performance. It prints the board state at
    each iteration to match the specified output format.
   
    10000 iterations give more chances to escape local optima and find global optima.
    And, allows for more extensive exploration of the solution space, 
    which can be beneficial for larger board sizes or more complex problems.
    """
    # Define the available search strategies
    strategies = [
        ("Hill Climbing", hill_climbing),
        ("Simulated Annealing", simulated_annealing),
        ("Genetic Algorithm", genetic_algorithm),
        ("Local Beam Search", local_beam_search)
    ]
    strategy_scores = {name: 1 for name, _ in strategies}  # Initialize scores for each strategy

    current_board = initialize_board(n)  # Generate an initial random board
    print(f"\nInitial board: {current_board}\n")  # Print the initial board

    for iteration in range(1, max_iterations + 1):  # Loop for the specified number of iterations
        print(f"Iteration {iteration}: {current_board}\n")  # Print the current board state

        # Select the strategy with the highest score
        strategy_name, strategy_func = max(strategies, key=lambda s: strategy_scores[s[0]])
        result_board, result_conflicts = strategy_func(current_board[:])  # Apply the selected strategy

        if result_conflicts == 0:  # If a solution is found
            return result_board  # Return the solution

        # Update strategy scores based on improvement
        improvement = calculate_conflicts(current_board) - result_conflicts
        strategy_scores[strategy_name] += improvement
        current_board = result_board  # Update the current board

        # Normalize scores to prevent one strategy from dominating
        total_score = sum(strategy_scores.values())
        strategy_scores = {name: score / total_score for name, score in strategy_scores.items()}

    return current_board  # Return the best board found after all iterations

def print_board(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[col] == row:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <board_size>")
        sys.exit(1)

    try:
        n = int(sys.argv[1])
        if n < 4:
            print("Board size must be at least 4")
            sys.exit(1)
    except ValueError:
        print("Board size must be an integer")
        sys.exit(1)
    #testing time limits
    start_time = time.time()
    solution = local_search(n)
    end_time = time.time()

    print(f"\nFinal solution for {n}-queens problem:")
    print(solution)
    if n <= 20:
        print_board(solution)
    print(f"Conflicts: {calculate_conflicts(solution)}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()