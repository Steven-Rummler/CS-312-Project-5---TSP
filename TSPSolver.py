#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))


import time
import numpy as np
from TSPClasses import *
import heapq as hq
import itertools
import copy


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    def makeNullResults(self, time_taken):
        results = {}
        results['cost'] = float('inf')
        results['time'] = time_taken
        results['count'] = None
        results['soln'] = None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time()-start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

    def greedy(self, time_allowance=60.0):
        # Start Timer
        start_time = time.time()

        # Initial Setup
        cities = self._scenario.getCities()
        n = len(cities)
        route = []
        distances = np.array([[cities[s].costTo(cities[d])
                               for d in range(n)] for s in range(n)])
        remaining = [i for i in range(1, n)]
        available = [[True for d in range(n)] for s in range(n)]

        # Greedy Depth-First Search
        while time.time()-start_time < time_allowance:
            # Handle start of route
            if len(route) == 0:
                # Find closest city
                min_city = None
                min_dist = float('inf')
                for city in remaining:
                    if distances[0][city] < min_dist and available[0][city]:
                        min_city = city
                        min_dist = distances[0][city]
                # If no route from start, there is no route
                if min_dist == float('inf'):
                    return self.makeNullResults(time.time() - start_time)
                # Travel to nearest city
                else:
                    route.append(min_city)
                    remaining.remove(min_city)
                    available[0][min_city] = False
                    # print('Traveling from', cities[0]._name, 'to', cities[min_city]._name)
            # Handle end of route
            elif len(remaining) == 0:
                # Backtrack from dead end
                if distances[route[-1]][0] == float('inf'):
                    for dest in range(n):
                        available[route[-1]][dest] = True
                    remaining.append(route[-1])
                    route = route[:-1]
                    # print('Backtracking from', cities[route[-1]]._name, 'to', cities[route[-2]]._name)
                # We're done, return solution
                else:
                    bssf = TSPSolution([cities[0]] + [cities[i]
                                       for i in route])
                    results = {}
                    results['cost'] = bssf.cost
                    results['time'] = time.time() - start_time
                    results['count'] = None
                    results['soln'] = bssf
                    results['max'] = None
                    results['total'] = None
                    results['pruned'] = None
                    return results
            # Handle intermediate step
            else:
                # Find the closest city
                current = route[-1]
                min_city = None
                min_dist = float('inf')
                for city in remaining:
                    if distances[current][city] < min_dist and available[current][city]:
                        min_city = city
                        min_dist = distances[current][city]
                # Backtrack from dead end
                if min_dist == float('inf'):
                    for dest in range(n):
                        available[route[-1]][dest] = True
                    remaining.append(route[-1])
                    route = route[:-1]
                    # print('Backtracking from', cities[current]._name, 'to', cities[previous]._name)
                # Continue to closest city
                else:
                    route.append(min_city)
                    remaining.remove(min_city)
                    available[current][min_city] = False
                    # print('Traveling from', cities[current]._name, 'to', cities[min_city]._name)
        return self.makeNullResults(time.time() - start_time)

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

    def branchAndBound(self, time_allowance=60.0):
        # Start Timer
        start_time = time.time()

        # Initial Setup
        results = {}
        cities = self._scenario.getCities()
        count = 0
        pruned_states = 0
        total_states = 1
        max_states = 0
        bssf = None
        states = []
        hq.heapify(states)

        # Get an Initial BSSF
        bssf = self.greedy(time_allowance)['soln']

        # Build first state
        start_grid = np.array([[cities[source].costTo(cities[destination]) for destination in range(
            len(cities))] for source in range(len(cities))])
        # np.set_printoptions(linewidth=400, precision=0)

        initial_state = State([0], start_grid, 0)
        hq.heappush(states, initial_state)

        # Branch and Bound
        while len(states) > 0 and time.time()-start_time < time_allowance:
            # Update max number of states in queue
            max_states = max_states if len(
                states) < max_states else len(states)

            # Get the most promising state from the queue
            state = hq.heappop(states)

            # Check if we need to prune the state
            if bssf is None or state.bound < bssf.cost:
                # For each destination except 0, make a new state
                state.remaining_destinations.remove(0)
                for city in state.remaining_destinations:
                    total_states += 1

                    # Add a new state with the new city added to the route and
                    # the cost of travelling there added to the lower bound
                    new_state = State(
                        state.route + [city], np.copy(state.grid), state.bound + state.grid[state.route[-1]][city])

                    # If the route includes all cities
                    if len(new_state.route) == len(cities):
                        # Add the final edge to the cost and first city to the route
                        new_state.bound += new_state.grid[new_state.route[-1]][0]

                        # If the route is valid, count it as a solution
                        if new_state.bound != float('inf'):
                            count += 1

                            # If the route is our best so far, update accordingly
                            if bssf is None or new_state.bound < bssf.cost:
                                bssf = TSPSolution([cities[i]
                                                    for i in new_state.route])
                            else:
                                # Prune it
                                pruned_states += 1
                    # If the new state still has potential, add it to the queue
                    elif bssf is None or new_state.bound < bssf.cost:
                        hq.heappush(states, new_state)
                    # Otherwise ignore it
            else:
                pruned_states += 1
                # print('Pruned route:', state.route)
        end_time = time.time()
        results['cost'] = bssf.cost if bssf is not None else math.inf
        results['time'] = end_time - start_time
        results['count'] = count - 1
        results['soln'] = bssf
        results['max'] = max_states
        results['total'] = total_states
        results['pruned'] = pruned_states
        return results

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

    def fancy(self, time_allowance=60.0):
        # Start Timer
        start_time = time.time()

        # Initial Setup
        greedy_results = self.greedy(time_allowance)
        if (greedy_results['cost'] == math.inf):
            return greedy_results
        greedy = greedy_results['soln']
        route = [city._index for city in greedy.route]
        cost = greedy.cost
        time_to_best = 0
        solutions_to_best = 0
        print('Greedy solution has cost:', cost)
        bssfs = 0
        routes = 0
        n = len(route)
        countdown = 1000*n
        indexes = range(n)
        cities = self._scenario.getCities()
        cost_lookup = np.array([[cities[s].costTo(cities[d])
                               for d in range(n)] for s in range(n)])

        # Local Search
        while countdown > 0 and time.time() - start_time < time_allowance:
            countdown -= 1
            new_cost = cost
            a, b = random.sample(indexes, 2)
            # Get city indexes
            a_city = route[a]
            b_city = route[b]
            a_prev = route[a-1]
            b_prev = route[b-1]
            a_next = route[0] if a + 1 >= n else route[a+1]
            b_next = route[0] if b + 1 >= n else route[b+1]
            # Subtract cost to/from a and b
            new_cost -= cost_lookup[a_prev][a_city]
            new_cost -= cost_lookup[b_prev][b_city]
            new_cost -= cost_lookup[a_city][a_next]
            new_cost -= cost_lookup[b_city][b_next]
            # Add cost to/from b and a
            new_cost += cost_lookup[a_prev][b_city]
            new_cost += cost_lookup[b_prev][a_city]
            new_cost += cost_lookup[a_city][b_next]
            new_cost += cost_lookup[b_city][a_next]
            # Check the route and make the switch if it's better
            routes += 1
            if (new_cost < cost):
                print('New route reduces cost by', cost-new_cost)
                route[a], route[b] = route[b], route[a]
                cost = new_cost
                countdown = 1000*n
                time_to_best = time.time() - start_time
                solutions_to_best = routes
                bssfs += 1

        # Finish and Return Results
        print('Time to Best:', time_to_best)
        print('Solutions to Best:', solutions_to_best)

        end_time = time.time()
        results = {}
        results['cost'] = cost
        results['time'] = end_time - start_time
        results['count'] = bssfs
        results['soln'] = TSPSolution([cities[city] for city in route])
        results['max'] = 1
        results['total'] = routes
        results['pruned'] = 0
        return results


class State:
    def __init__(self, route, grid, bound):
        self.route = route
        self.grid = grid
        self.bound = bound

        # The remaining places from which we can travel are the places
        # we haven't visited, plus the city we are currently at.
        self.remaining_sources = [city for city in range(
            len(grid)) if city not in route] + [route[-1]]

        # The remaining places to which we can travel are the places
        # we haven't visited, plus the city at position 0
        self.remaining_destinations = [0] + [city for city in range(
            len(grid)) if city not in route]

        # Reduce the outgoing paths from each remaining source.
        if self.bound != float('inf'):
            for row in self.remaining_sources:
                minimum_value = float('inf')
                for col in self.remaining_destinations:
                    if (self.grid[row][col] < minimum_value):
                        minimum_value = self.grid[row][col]
                if minimum_value == float('inf'):
                    self.bound = float('inf')
                    break
                elif minimum_value != 0:
                    self.bound += minimum_value
                    for col in self.remaining_destinations:
                        self.grid[row][col] -= minimum_value

        # Reduce the incoming paths to each remaining destination
        if self.bound != float('inf'):
            for col in self.remaining_destinations:
                minimum_value = float('inf')
                for row in self.remaining_sources:
                    if (self.grid[row][col] < minimum_value):
                        minimum_value = self.grid[row][col]
                if minimum_value == float('inf'):
                    self.bound = float('inf')
                    break
                elif minimum_value != 0:
                    self.bound += minimum_value
                    for row in self.remaining_sources:
                        self.grid[row][col] -= minimum_value

        # Calculate an adjusted score to prioritize deeper parts of the tree
        self.score = self.bound * len(self.grid) / len(self.route)

    def __lt__(self, other):
        return self.score < other.score

# grid = np.array([[float('inf'), 1, 2], [3, float('inf'), 4], [5, 6, float('inf')]])
# route = []
# bound = 0

# state = State(route, grid, bound)

# print('route\n', state.route)
# np.set_printoptions(linewidth=400, precision=0)
# print('grid\n', state.grid)
# print('bound\n', state.bound)
