import random

class TreeNode:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.children = {}
        self.value = 0

    def add_child(self, action, child):
        self.children[action] = child

    def add_parent(self, parent):
        self.parent = parent

class MonteCarloTreeSearch:
  def __init__(self, game, max_iterations):
    self.game = game
    self.max_iterations = max_iterations

  def search(self, state):
    # Initialize the tree with the current state of the game.
    root = TreeNode(state)

    # Iterate for a given number of iterations.
    for i in range(self.max_iterations):
      # Select a node to expand using the UCB1 algorithm.
      selected_node = self.select_node(root)

      # Expand the selected node by simulating a random playout.
      result = self.simulate_playout(selected_node.state)

      # Update the selected node with the result of the playout.
      self.backpropagate(selected_node, result)

    # Return the move with the highest number of wins.
    return self.get_best_move(root)

  def select_node(self, node):
    # If the node is a leaf node, return it.
    if node.is_leaf():
      return node

    # Calculate the UCB1 score for each child node.
    scores = [self.ucb1(node, child) for child in node.children.values()]

    # Return the child node with the highest UCB1 score.
    return node.children[scores.index(max(scores))]

  def simulate_playout(self, state):
    # Make a copy of the current state.
    current_state = state.copy()

    # Play the game randomly until it is finished.
    while not self.game.is_goal(current_state):
      # Get a list of valid successor states.
      successors = self.game.get_successors(current_state)

      # Choose a random successor state.
      current_state = random.choice(successors)

    # Return the result of the game.
    return self.game.is_goal(current_state)
