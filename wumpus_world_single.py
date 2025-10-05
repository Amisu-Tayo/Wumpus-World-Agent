# Note: AI assistance from Google's Gemini was used for debugging and algorithm refinement.
from __future__ import annotations

import sys, math, random
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Tuple, Set, List, Dict, Optional, Literal, TypedDict

Action = Literal['forward', 'left', 'right', 'grab', 'shoot', 'climb']
class StepInfo(TypedDict, total = False):
    reason: str
    death_cell: "Coordinate"
    climbed: bool

import pygame

# ------------------------- Config -------------------------
GRID_SIZE = 4
PIT_PROB = 0.20
SEED = None                 # set an int for reproducible runs
ANIMATION_DELAY_MS = 400
MAX_STEPS = 220

# ------------------------- Types --------------------------
Coordinate = Tuple[int, int]

@dataclass(frozen=True)
class Percept:
    breeze: bool
    stench: bool
    glitter: bool
    bump: bool
    scream: bool

@dataclass(frozen=True)
class StepResult:
    percept: Percept
    reward: int
    done: bool
    info: StepInfo

# ------------------------- Environment --------------------
class WumpusWorld:
    E, N, W, S = 0, 1, 2, 3
    DIRS = {E: (0, 1), N: (-1, 0), W: (0, -1), S: (1, 0)}

    def __init__(self, n=4, pit_prob=0.2, seed=None):
        self.n = n
        self.pit_prob = pit_prob
        self.rng = random.Random(seed)
        self.reset()

    # --- helpers ---
    def _cells(self):
        for r in range(1, self.n + 1):
            for c in range(1, self.n + 1):
                yield (r, c)

    def _neighbors(self, cell: Coordinate):
        r, c = cell
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 1 <= nr <= self.n and 1 <= nc <= self.n:
                yield (nr, nc)

    # --- layout ---
    def _generate_layout(self):
        start = (1, 1)
        self.pits: Set[Coordinate] = set()
        # pits everywhere except start with PIT_PROB
        for cell in self._cells():
            if cell == start: 
                continue
            if self.rng.random() < self.pit_prob:
                self.pits.add(cell)

        # Wumpus anywhere except start 
        wumpus_candidates = [c for c in self._cells() if c != start]
        self.wumpus: Optional[Coordinate] = self.rng.choice(wumpus_candidates)
        self.wumpus_alive = True

        # Gold anywhere except start (may be co-located with pit or Wumpust)
        gold_candidates = [c for c in self._cells() if c != start]
        self.gold: Optional[Coordinate] = self.rng.choice(gold_candidates)

    def reset(self):
        self.score = 0
        self.bump = False
        self.scream = False
        self.alive = True
        self.has_gold = False
        self.arrow_available = True

        self._generate_layout()

        self.pos: Coordinate = (1, 1)
        self.dir = WumpusWorld.E
        self.visited: Set[Coordinate] = {(1, 1)}
        return self._observe()

    # --- percept field generators for UI ---
    def get_breeze_cells(self) -> Set[Coordinate]:
        breeze_cells = set()
        for cell in self._cells():
            if any(nb in self.pits for nb in self._neighbors(cell)):
                breeze_cells.add(cell)
        return breeze_cells

    def get_stench_cells(self) -> Set[Coordinate]:
        stench_cells = set()
        if self.wumpus and self.wumpus_alive:
            for cell in self._cells():
                if any(nb == self.wumpus for nb in self._neighbors(cell)):
                    stench_cells.add(cell)
        return stench_cells

    # --- core mechanics ---
    def step(self, action: Action) -> StepResult:
        assert action in {'forward','left','right','grab','shoot','climb'}
        reward = -1
        self.bump = False
        self.scream = False

        if not self.alive:
            return StepResult(self._observe(), reward, True, {'reason': 'GAME OVER: Agent already dead.'})

        if action == 'left':
            self.dir = (self.dir + 1) % 4

        elif action == 'right':
            self.dir = (self.dir - 1 + 4) % 4

        elif action == 'forward':
            dr, dc = WumpusWorld.DIRS[self.dir]
            nr, nc = self.pos[0] + dr, self.pos[1] + dc
            if not (1 <= nr <= self.n and 1 <= nc <= self.n):
                self.bump = True
            else:
                self.pos = (nr, nc)
                self.visited.add(self.pos)
                # death check (pit or live Wumpus)
                if (self.pos in self.pits) or (self.wumpus_alive and self.pos == self.wumpus):
                    self.alive = False
                    reward += -1000
                    return StepResult(self._observe(), reward, True, {'reason': 'GAME OVER: Killed by a hazard'})

        elif action == 'grab':
            if self.gold and self.pos == self.gold and not self.has_gold:
                self.has_gold = True
                self.gold = None

        elif action == 'shoot':
            if self.arrow_available:
                self.arrow_available = False
                reward += -10
                self._shoot_arrow()

        elif action == 'climb':
            if self.pos == (1,1):
                msg = 'SUCCESS: Climbed out with gold!' if self.has_gold else 'GAME OVER: Climbed out without gold.'
                if self.has_gold:
                    reward += 1000
                return StepResult(self._observe(), reward, True, {'reason': msg})

        return StepResult(self._observe(), reward, False, {'reason': ''})

    def _shoot_arrow(self):
        # Arrow starts in the NEXT cell in facing direction
        r, c = self.pos
        dr, dc = WumpusWorld.DIRS[self.dir]
        r += dr
        c += dc
        while 1 <= r <= self.n and 1 <= c <= self.n:
            if self.wumpus_alive and (r, c) == self.wumpus:
                self.wumpus_alive = False
                self.scream = True
                break
            r += dr
            c += dc

    def _observe(self) -> Percept:
        breeze = any(nb in self.pits for nb in self._neighbors(self.pos))
        stench = self.wumpus_alive and any(nb == self.wumpus for nb in self._neighbors(self.pos)) if self.wumpus else False
        glitter = (self.gold is not None and self.pos == self.gold)
        return Percept(breeze=breeze, stench=stench, glitter=glitter, bump=self.bump, scream=self.scream)

    # for drawing
    def get_layout(self):
        return {
            'n': self.n,
            'pits': set(self.pits),
            'wumpus': self.wumpus,
            'wumpus_alive': self.wumpus_alive,
            'gold': self.gold,
            'agent_pos': self.pos,
            'agent_dir': self.dir,
            'has_gold': self.has_gold,
            'visited': set(self.visited),
            'breeze_cells': self.get_breeze_cells(),
            'stench_cells': self.get_stench_cells()
        }

# ------------------------- Agent (Rule + BFS + EV) -----------------------
class HybridAgent:
    """
    Score-maximizing agent for a small grid:
      - Deterministic inference: safe propagation, pit singles from breezes,
        Wumpus triangulation via stench intersections + no-stench eliminations.
      - BFS on (safe | visited) to frontiers/home.
      - Expected-Value (EV) choice among: go home now, explore frontier, or guess
        the least-risk unknown neighbor (only when NOT holding gold).
      - After grabbing gold: never guess; only traverse known-safe to (1,1).
      - Arrow: shoot only if Wumpus location is confirmed & aligned and shooting is useful.
    """
    def __init__(self, grid_size=4, pit_prob=0.2):
        self.n = grid_size
        self.reset_episode()

    # ---------- lifecycle ----------
    def reset_episode(self):
        self.pos = (1, 1)
        self.dir = WumpusWorld.E
        self.steps = 0

        # UI-facing flags
        self.visited: Set[Coordinate] = {(1,1)}
        self.safe: Set[Coordinate] = {(1,1)}      # known safe
        self.wumpus_loc_confirmed: Optional[Coordinate] = None
        self.wumpus_dead_known = False
        self.has_gold = False
        self.arrow = True

        # Knowledge
        self.percepts: Dict[Coordinate, Percept] = {}
        self.pit_votes: Dict[Coordinate, int] = defaultdict(int)
        self.wumpus_votes: Dict[Coordinate, int] = defaultdict(int)
        self.def_pits: Set[Coordinate] = set()
        self.cand_wumpus: Set[Coordinate] = set()
        self.def_wumpus: Optional[Coordinate] = None

        # control
        self._glitter_now = False
        self._last_bump = False
        self._last_plan_actions: deque[str] = deque()

    # ---------- utilities ----------
    @staticmethod
    def _neighbors(cell: Coordinate, n: int):
        r, c = cell
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 1 <= nr <= n and 1 <= nc <= n:
                yield (nr, nc)

    def _adj(self, cell):  # convenience
        return list(self._neighbors(cell, self.n))

    def _bfs_path(self, start: Coordinate, goal: Coordinate, passable: Set[Coordinate]) -> Optional[List[Coordinate]]:
        if start == goal: return [start]
        q = deque([start]); parent = {start: None}
        while q:
            u = q.popleft()
            for v in self._neighbors(u, self.n):
                if v in passable and v not in parent:
                    parent[v] = u
                    if v == goal:
                        path=[v]
                        while parent[path[-1]] is not None:
                            path.append(parent[path[-1]])
                        return list(reversed(path))
                    q.append(v)
        return None

    def _convert_coords_to_actions(self, current_pos: Coordinate, current_dir: int, path: List[Coordinate]) -> List[Action]:
        def rotations_to_face(target: Coordinate, pos: Coordinate, facing: int) -> List[str]:
            tr, tc = target; r, c = pos
            if tr == r and tc == c + 1: desired = 0  # E
            elif tr == r - 1 and tc == c: desired = 1  # N
            elif tr == r and tc == c - 1: desired = 2  # W
            elif tr == r + 1 and tc == c: desired = 3  # S
            else: return []
            actions = []; diff = (facing - desired + 4) % 4
            if diff == 1: actions.append('right')
            elif diff == 2: actions.extend(['right','right'])
            elif diff == 3: actions.append('left')
            return actions

        actions = []; pos = current_pos; facing = current_dir
        for nxt in path[1:]:
            rots = rotations_to_face(nxt, pos, facing)
            for r in rots:
                actions.append(r)
                facing = (facing + 1) % 4 if r == 'left' else (facing - 1 + 4) % 4
            actions.append('forward')
            pos = nxt
        return actions

    # ---------- inference ----------
    def _propagate_safety(self):
        changed = False
        for c,p in self.percepts.items():
            if not p.breeze and (not p.stench or self.wumpus_dead_known):
                for nb in self._adj(c):
                    if nb not in self.safe and nb not in self.def_pits:
                        self.safe.add(nb); changed = True
        return changed

    def _deduce_from_breeze(self):
        changed = False
        for c,p in self.percepts.items():
            if not p.breeze: 
                continue
            unknown = [nb for nb in self._adj(c) if nb not in self.safe and nb not in self.def_pits]
            if len(unknown) == 1:
                pit = unknown[0]
                if pit not in self.def_pits:
                    self.def_pits.add(pit); changed = True
        return changed
    
    def _deduce_pits_by_intersection(self):
        changed = False
        breeze_cells = [c for c,p in self.percepts.items() if p.breeze]
        if len(breeze_cells) < 2:
            return False
            
        
        # consider every pair of breeze cells
        for i in range(len(breeze_cells)):
            for j in range(i+1, len(breeze_cells)):
                c1, c2 = breeze_cells[i], breeze_cells[j]
                nb1 = {nb for nb in self._adj(c1) if nb not in self.safe and nb not in self.def_pits}
                nb2 = {nb for nb in self._adj(c2) if nb not in self.safe and nb not in self.def_pits}
                intersection = nb1 & nb2
                if len(intersection) == 1:
                    pit = next(iter(intersection))
                    if pit not in self.def_pits:
                        self.def_pits.add(pit); changed = True
        return changed       

    def _recompute_wumpus_candidates(self):
        if self.wumpus_dead_known:
            self.cand_wumpus.clear(); self.def_wumpus = None
            self.wumpus_loc_confirmed = None
            return
        stench_cells = [c for c,p in self.percepts.items() if p.stench]
        if not stench_cells:
            self.cand_wumpus.clear(); self.def_wumpus = None; self.wumpus_loc_confirmed = None
            return
        cand = {nb for nb in self._adj(stench_cells[0]) if nb not in self.safe and nb not in self.def_pits}
        for c in stench_cells[1:]:
            nbset = {nb for nb in self._adj(c) if nb not in self.safe and nb not in self.def_pits}
            cand &= nbset
        # no-stench eliminations
        for c,p in self.percepts.items():
            if not p.stench:
                for nb in self._adj(c):
                    cand.discard(nb)
        self.cand_wumpus = cand
        self.def_wumpus = next(iter(cand)) if len(cand) == 1 else None
        self.wumpus_loc_confirmed = self.def_wumpus

    def _iterate_inference(self):
        while True:
            changed = False
            changed |= self._propagate_safety()
            changed |= self._deduce_from_breeze()
            changed |= self._deduce_pits_by_intersection()
            self._recompute_wumpus_candidates()
            if not changed: break

    # ---------- risk & EV ----------
    def _risk_score(self, cell: Coordinate) -> float:
        if cell in self.def_pits: return 1.0
        if not self.wumpus_dead_known and self.def_wumpus == cell: return 1.0
        pit_p = 1 - (0.5 ** self.pit_votes[cell]) if self.pit_votes[cell] else 0.0
        wum_p = 0.0
        if not self.wumpus_dead_known:
            base = 0.6 if cell in self.cand_wumpus else 0.0
            wum_p = max(base, 1 - (0.5 ** self.wumpus_votes[cell]) if self.wumpus_votes[cell] else base)
        return 1 - (1 - pit_p) * (1 - wum_p)

    def _gold_prior(self):
        unknown = []
        for r in range(1, self.n+1):
            for c in range(1, self.n+1):
                cell = (r,c)
                if cell in self.safe or cell in self.visited or cell in self.def_pits:
                    continue
                unknown.append(cell)
        return (1/len(unknown) if unknown else 0.0), unknown

    # ---------- arrow logic ----------
    def _maybe_plan_shot(self) -> Optional[str]:
        if self.wumpus_dead_known or self.def_wumpus is None or not self.arrow:
            return None
        wr, wc = self.def_wumpus
        r, c = self.pos
        # must be aligned
        if r != wr and c != wc:
            return None
        # rotate toward target
        desired_dir = None
        if r == wr:
            desired_dir = WumpusWorld.E if wc > c else WumpusWorld.W
        else:
            desired_dir = WumpusWorld.S if wr > r else WumpusWorld.N
        if self.dir != desired_dir:
            diff = (self.dir - desired_dir + 4) % 4
            return 'right' if diff in (1,2) else 'left'
        # Shoot (EV heuristic: shooting is allowed here; −10 is often worth the safety)
        return 'shoot'

    # ---------- I/O with game loop ----------
    def update_with_percept(self, percept: Percept):
        self.percepts[self.pos] = percept
        self._last_bump = percept.bump
        self._glitter_now = percept.glitter
        if percept.scream:
            self.wumpus_dead_known = True
            self.cand_wumpus.clear()
            self.def_wumpus = None
            self.wumpus_loc_confirmed = None

        # votes (soft info)
        if percept.breeze:
            for nb in self._adj(self.pos):
                if nb not in self.safe:
                    self.pit_votes[nb] += 1
        if percept.stench and not self.wumpus_dead_known:
            for nb in self._adj(self.pos):
                if nb not in self.safe:
                    self.wumpus_votes[nb] += 1

        # deterministic inference
        self._iterate_inference()

    def _frontiers(self) -> List[Coordinate]:
        res = []
        for s in self.safe:
            for nb in self._adj(s):
                if nb not in self.visited and nb not in self.def_pits:
                    res.append(s); break
        return res

    def _least_risk_unvisited_neighbor(self) -> Optional[Coordinate]:
        best = None; best_score = 1e9
        for nb in self._adj(self.pos):
            if nb in self.visited or nb in self.def_pits:
                continue
            score = self._risk_score(nb)
            if score < best_score:
                best_score = score; best = nb
        return best

    def choose_action(self, percept: Percept) -> Action:
        if self._last_plan_actions:
            return self._last_plan_actions.popleft()
        
        self.steps += 1

        # Immediate actions
        if self._glitter_now and not self.has_gold:
            return 'grab'
        if self.has_gold and self.pos == (1,1):
            return 'climb'

        
        if self.has_gold:
            path_home = self._bfs_path(self.pos, (1,1), passable=(self.safe - self.def_pits))
            if path_home and len(path_home) > 1:
                self._last_plan_actions = deque(self._convert_coords_to_actions(self.pos, self.dir, path_home))
                return self._last_plan_actions.popleft()
            shot = self._maybe_plan_shot()
            if shot: return shot
            return 'right'  # harmless spin; we won't yolo into unknown with gold

        # Opportunistic safe shot before planning (if really sure)
        if not self.wumpus_dead_known and self.def_wumpus is not None and self.arrow:
            shot = self._maybe_plan_shot()
            if shot: return shot

        # Plan selection by EV: Go Home vs Frontier vs Guess
        passable = (self.safe | self.visited) - self.def_pits

        # EV_home
        path_h = self._bfs_path(self.pos, (1,1), passable=passable)
        dist_home = (len(path_h)-1) if path_h else 0
        EV_home = -dist_home

        # EV_frontier
        EV_frontier = -1e9; best_frontier_path = None
        fronts = self._frontiers()
        if fronts:
            best_path = None
            for f in fronts:
                p = self._bfs_path(self.pos, f, passable)
                if p and (best_path is None or len(p) < len(best_path)):
                    best_path = p
            if best_path:
                p_gold, _unknown = self._gold_prior()
                cost_to_F = len(best_path) - 1
                cost_back = len(self._bfs_path(best_path[-1], (1,1), passable) or []) - 1
                peek_cost = 1
                EV_frontier = -cost_to_F - peek_cost + p_gold * (1000 - max(cost_back,0))
                best_frontier_path = best_path

        # EV_guess (only when not holding gold)
        EV_guess = -1e9; best_guess = None
        cand = []
        for nb in self._adj(self.pos):
            if nb in self.visited or nb in self.safe or nb in self.def_pits: 
                continue
            cand.append((self._risk_score(nb), nb))
        if cand:
            cand.sort()
            risk, g = cand[0]
            p_gold, unknown = self._gold_prior()
            cost_back_from_g = len(self._bfs_path(g, (1,1), passable) or []) - 1
            EV_if_safe = -1 + p_gold * (1000 - max(cost_back_from_g,0))
            EV_guess = (1 - risk) * EV_if_safe + risk * (-1000)
            best_guess = g
        # ---- Choose best (availability-aware with clear tie-breaker) ----

        # Make EVs availability-aware: if an option isn't executable now, set to -inf
        home_available = (self.pos == (1,1)) or (path_h is not None and dist_home > 0)
        frontier_available = best_frontier_path is not None
        guess_available = best_guess is not None

        ev_home_eff = EV_home if home_available else float("-inf")
        ev_frontier_eff = EV_frontier if frontier_available else float("-inf")
        ev_guess_eff = EV_guess if guess_available else float("-inf")

        # Tie-break priority: Home (0) > Frontier (1) > Guess (2)
        candidates = [
            ("home", ev_home_eff, 0),
            ("frontier", ev_frontier_eff, 1),
            ("guess", ev_guess_eff, 2),
        ]

        # Pick by EV first, then by priority (lower is better)
        best_label, _, _ = max(candidates, key=lambda t: (t[1], -t[2]))

        if best_label == "home":
            if self.pos == (1,1):
                return 'climb'
            # We know path_h exists and has length > 1 if home_available and not at start
            acts = self._convert_coords_to_actions(self.pos, self.dir, path_h)
            self._last_plan_actions = deque(acts)
            return self._last_plan_actions.popleft()


        elif best_label == "frontier":
            # If we need to travel to a frontier tile, do so.
            if best_frontier_path and len(best_frontier_path) > 1:
                acts = self._convert_coords_to_actions(self.pos, self.dir, best_frontier_path)
                self._last_plan_actions = deque(acts)
                return self._last_plan_actions.popleft()
            # We are already standing on the frontier tile (common at start).
            nb = self._least_risk_unvisited_neighbor()
            if nb is not None:
                acts = self._convert_coords_to_actions(self.pos, self.dir, [self.pos, nb])
                if acts:
                    self._last_plan_actions = deque(acts)
                    return self._last_plan_actions.popleft()
                # If we're already facing it, acts may be just ['forward'] anyway; but guard just in case.
                return 'forward'
            # No unknown neighbors? rotate to find a new frontier next tick.
            return 'right'


        elif best_label == "guess":
            acts = self._convert_coords_to_actions(self.pos, self.dir, [self.pos, best_guess])
            self._last_plan_actions = deque(acts)
            return self._last_plan_actions.popleft()

        # Last-resort fallback if literally nothing is available
        if self._last_bump:
            return 'right'
        return 'forward'
    
    def update_state_after_action(self, action: str):
        if action == 'grab':
            self.has_gold = True
        if action == 'shoot':
            self.arrow = False
        self.visited.add(self.pos)

# ------------------------- Drawing ------------------------
CELL = 100
MARGIN = 40
SIDEBAR = 260
SMALL_FONT_SIZE = 18
BG = (245,245,245)
INFO_BG = (245,248,252)

def draw_wumpus(surface, rect, alive, is_ghost=False):
    color = (138, 43, 226)
    if is_ghost:
        s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        s.fill((color[0], color[1], color[2], 90))
        surface.blit(s, rect.topleft); return
    pygame.draw.rect(surface, color, rect.inflate(-20, -20))
    eye_color = (255, 255, 0)
    eye1 = pygame.Rect(rect.centerx - 15, rect.centery - 10, 8, 8)
    eye2 = pygame.Rect(rect.centerx + 7, rect.centery - 10, 8, 8)
    pygame.draw.ellipse(surface, eye_color, eye1)
    pygame.draw.ellipse(surface, eye_color, eye2)
    if not alive:
        for e in (eye1, eye2):
            pygame.draw.line(surface, (0,0,0), e.topleft, e.bottomright, 2)
            pygame.draw.line(surface, (0,0,0), e.topright, e.bottomleft, 2)

def draw_agent(surface, center, facing, has_gold):
    cx, cy = center; size = 24
    ang_map = {0: 0, 1: -90, 2: 180, 3: 90}
    ang = math.radians(ang_map.get(facing, 0)); pts = []
    for a in [0, 140, -140]:
        aa = ang + math.radians(a); r = size if a == 0 else size * 0.55
        pts.append((cx + r * math.cos(aa), cy + r * math.sin(aa)))
    pygame.draw.polygon(surface, (30,144,255), pts)
    pygame.draw.polygon(surface, (0,80,160), pts, 2)
    if has_gold:
        gold_rect = pygame.Rect(cx-5, cy-5, 10, 10)
        pygame.draw.rect(surface, (255,215,0), gold_rect)
        pygame.draw.rect(surface, (0,0,0), gold_rect, 1)

def draw_world(screen, env: WumpusWorld, agent: HybridAgent, font, small_font, action_log, total=0, effects=None):
    if effects is None: effects = {}
    screen.fill(BG)
    layout = env.get_layout(); n = layout['n']
    grid_w = MARGIN * 2 + CELL * n
    offset_x = offset_y = 0

    # draw grid
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            cell = (r, c)
            x = MARGIN + (c - 1) * CELL
            y = MARGIN + (n - r) * CELL
            rect = pygame.Rect(x, y, CELL, CELL)

            # bg
            if cell == (1,1):
                pygame.draw.rect(screen, (220,235,255), rect)
            else:
                fill = (255,255,255)
                if cell in agent.visited: fill = (220,220,220)
                if cell in agent.safe and cell not in agent.visited: fill = (200,255,200)
                pygame.draw.rect(screen, fill, rect)

            # hazards & items
            if cell in layout['pits']:
                pygame.draw.circle(screen, (0,0,0), rect.center, CELL // 3)
            if layout['wumpus'] == cell and layout['wumpus_alive']:
                draw_wumpus(screen, rect, True)
            if layout['gold'] == cell:
                g = font.render("G", True, (255,215,0))
                screen.blit(g, g.get_rect(center=rect.center))

            # percept overlays
            if cell in layout['breeze_cells']:
                b = small_font.render("B", True, (0,120,255))
                screen.blit(b, (rect.x + 6, rect.y + 6))
            if cell in layout['stench_cells']:
                s = small_font.render("S", True, (255,120,0))
                screen.blit(s, (rect.right - 16, rect.y + 6))

            # ghost Wumpus speculation (optional)
            if agent.wumpus_loc_confirmed and not agent.wumpus_dead_known and cell == agent.wumpus_loc_confirmed:
                draw_wumpus(screen, rect, True, is_ghost=True)

            pygame.draw.rect(screen, (150,150,150), rect, 1)

            if cell == (1,1):
                start_surf = small_font.render("START", True, (0,0,100))
                screen.blit(start_surf, (x+5, y+5))

    # agent
    ar, ac = layout['agent_pos']
    ax = MARGIN + (ac - 1) * CELL + CELL // 2
    ay = MARGIN + (n - ar) * CELL + CELL // 2
    draw_agent(screen, (ax, ay), layout['agent_dir'], agent.has_gold)

    # sidebar
    sidebar_rect = pygame.Rect(grid_w, 0, SIDEBAR, MARGIN * 2 + CELL * n)
    pygame.draw.rect(screen, INFO_BG, sidebar_rect)
    pygame.draw.rect(screen, (180,190,200), sidebar_rect, 1)
    line = MARGIN // 2

    def put(txt, fnt=small_font, color=(0,0,0)):
        nonlocal line
        t = fnt.render(txt, True, color)
        screen.blit(t, (grid_w + 16, line))
        line += fnt.get_height() + 6

    put(f"Score: {total}", font)
    put(f"Steps: {agent.steps}")
    put(f"Carrying Gold: {'Yes' if agent.has_gold else 'No'}")
    put(f"Arrow: {'Available' if agent.arrow else 'Used'}")
    put(f"Wumpus Known Dead: {'Yes' if agent.wumpus_dead_known else 'No'}")
    put(f"Wumpus Confirmed: {agent.wumpus_loc_confirmed if agent.wumpus_loc_confirmed else 'No'}")
    line += 8
    put("Last Actions:")
    for a in action_log:
        put(f"- {a}")
    line += 10
    put("Legend:")
    put("B: Breeze (adjacent to pits)")
    put("S: Stench (adjacent to Wumpus)")

# ------------------------- Game Loop ----------------------
def run_game():
    pygame.init()
    pygame.display.set_caption("Wumpus World — Rule + BFS + EV Agent")
    font = pygame.font.SysFont(None, 42)
    small_font = pygame.font.SysFont(None, SMALL_FONT_SIZE)
    n = GRID_SIZE

    screen_width = MARGIN * 2 + CELL * n + SIDEBAR
    screen_height = MARGIN * 2 + CELL * n
    screen = pygame.display.set_mode((screen_width, screen_height))

    env = WumpusWorld(n=n, pit_prob=PIT_PROB, seed=SEED)
    agent = HybridAgent(grid_size=n, pit_prob=PIT_PROB)
    percept = env.reset()

    clock = pygame.time.Clock()
    total_score = 0
    action_log = deque(maxlen=6)
    paused = False
    game_over = False
    game_over_reason = ""
    animation_delay = ANIMATION_DELAY_MS

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p: paused = not paused
                if event.key == pygame.K_SPACE and game_over:
                    percept = env.reset(); agent.reset_episode()
                    total_score = 0; action_log.clear(); game_over = False
                if event.key == pygame.K_UP: animation_delay = max(0, animation_delay - 50)
                if event.key == pygame.K_DOWN: animation_delay += 50

        if not game_over:
            if not paused:
                agent.update_with_percept(percept)
                action = agent.choose_action(percept)
                action_log.append(action)

                step = env.step(action)
                percept = step.percept
                total_score += step.reward

                # sync agent with env
                agent.pos = env.pos
                agent.dir = env.dir
                agent.update_state_after_action(action)

                if step.done or agent.steps >= MAX_STEPS:
                    game_over = True
                    game_over_reason = step.info.get('reason', f'GAME OVER: Timed out after {MAX_STEPS} steps.')

        # draw
        draw_world(screen, env, agent, font, small_font, action_log, total_score)
        if paused:
            pause_surf = font.render("PAUSED", True, (255,0,0))
            screen.blit(pause_surf, pause_surf.get_rect(center=(screen_width - SIDEBAR//2, screen_height//2)))
        if game_over:
            overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0,0,0,160)); screen.blit(overlay, (0,0))
            t1 = font.render(game_over_reason.split(":")[0], True, (255,255,255))
            t2 = small_font.render(":".join(game_over_reason.split(":")[1:]).strip(), True, (230,230,230))
            t3 = font.render(f"Final Score: {total_score}", True, (255,255,0))
            t4 = small_font.render("Press SPACE to play again", True, (220,220,220))
            cy = screen_height//2
            screen.blit(t1, t1.get_rect(center=(screen_width//2, cy-60)))
            screen.blit(t2, t2.get_rect(center=(screen_width//2, cy-20)))
            screen.blit(t3, t3.get_rect(center=(screen_width//2, cy+25)))
            screen.blit(t4, t4.get_rect(center=(screen_width//2, cy+60)))

        pygame.display.flip()
        pygame.time.delay(animation_delay if not paused and not game_over else 0)
        clock.tick(60)

if __name__ == "__main__":
    try:
        run_game()
    except Exception:
        import traceback
        traceback.print_exc()
        input("Press ENTER to exit...")