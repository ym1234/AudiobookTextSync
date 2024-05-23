from queue import PriorityQueue
from tqdm import tqdm

indel = -1
match = 1
mismatch = -1


def heuristic(cur, goal):
    curdiag = cur[1] - cur[0]
    goaldiag = goal[1] - goal[0]
    ind = abs(curdiag - goaldiag)

    m = max(goal[0] - cur[0], goal[1] - cur[1]) - ind
    return match * m + ind * indel

def add_state(frontier, came_from, cost_so_far, score, prev, pos, goal):
    if cost_so_far.get(pos, -float('inf')) <= score:
        came_from[pos] = prev
        cost_so_far[pos] = score
        frontier.put((-score - heuristic(pos, goal), *pos))

def astar(a, b):
    goal = (len(a), len(b)) # (0,)*len(start):
    tqdm.write(str(goal))
    start = (0, 0)

    frontier = PriorityQueue()
    frontier.put((-heuristic(start, goal), *start))

    came_from = {}
    cost_so_far = {}

    came_from[start] = None
    cost_so_far[start] = 0

    pop = 0

    while not frontier.empty():
        tqdm.write("\rPOP COUNT: " +  str(pop), end='')
        # print("\rPOP COUNT: " +  str(pop), end='')
        score, ai, bi = frontier.get()
        score = -score - heuristic((ai, bi), goal)
        pop += 1

        if (ai, bi) == goal:
            # print(len(cost_so_far))
            cur = (ai, bi)
            while cur is not None:
                print(cur, used_move[cur] if cur in used_move else "")
                cur = came_from[cur]
            # break
            tqdm.write("\rPOP COUNT: " +  str(pop))
            return cost_so_far[goal]


        if ai < len(a):
            add_state(frontier, came_from, cost_so_far, score+indel, (ai, bi), (ai+1, bi), goal)

        if bi < len(b):
            add_state(frontier, came_from, cost_so_far, score+indel, (ai, bi), (ai, bi+1), goal)

        if ai < len(a) and bi < len(b):
            s = match if a[ai] == b[bi] else  mismatch
            add_state(frontier, came_from, cost_so_far, score+s, (ai, bi), (ai+1, bi+1), goal)
    tqdm.write("POP COUNT: " +  str(pop))
    return cost_so_far[goal]

print(astar("GCATGCU", "GATTACA"))
