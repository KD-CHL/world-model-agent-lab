"""Shared physical scene and conservative planar navigation map, in metres."""
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Box:
    name: str
    x: float
    y: float
    hx: float
    hy: float
    height: float
    color: tuple = (.45, .5, .55, 1.)


class IndoorScene:
    """Known-map indoor benchmark; furniture footprints are conservatively solid."""
    bounds = (-1.5, 6., -3., 3.)
    robot_radius = .4

    def __init__(self):
        self.boxes = (
            Box('island', 2., 0., .45, .65, .8, (.5, .3, .15, 1.)),
            Box('shelf', 4.8, 1.9, .35, .7, 1.6),
            Box('cabinet', 4.5, -2., .7, .3, 1.1),
            Box('wall_w', -1.6, 0., .1, 3.2, 1.8),
            Box('wall_e', 6.1, 0., .1, 3.2, 1.8),
            Box('wall_n', 2.25, 3.1, 3.85, .1, 1.8),
            Box('wall_s', 2.25, -3.1, 3.85, .1, 1.8),
        )

    def segment_free(self, a, b):
        r = self.robot_radius
        xmin, xmax, ymin, ymax = self.bounds
        if any(not (xmin+r <= p[0] <= xmax-r and ymin+r <= p[1] <= ymax-r) for p in (a, b)):
            return False
        for box in self.boxes:
            lo, hi = 0., 1.
            for start, end, center, half in ((a[0], b[0], box.x, box.hx+r),
                                             (a[1], b[1], box.y, box.hy+r)):
                d = end - start
                if abs(d) < 1e-12:
                    if not center-half <= start <= center+half:
                        lo, hi = 1., 0.
                        break
                else:
                    t1, t2 = (center-half-start)/d, (center+half-start)/d
                    lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            if lo <= hi:
                return False
        return True

    def route(self, start, goal, resolution=.2):
        if not self.segment_free(start, start) or not self.segment_free(goal, goal):
            raise ValueError('Start or goal is inside inflated obstacle/boundary')
        if self.segment_free(start, goal):
            return [start, goal]
        xmin, xmax, ymin, ymax = self.bounds
        nodes = [(xmin+i*resolution, ymin+j*resolution)
                 for i in range(round((xmax-xmin)/resolution)+1)
                 for j in range(round((ymax-ymin)/resolution)+1)]
        nodes = [p for p in nodes if self.segment_free(p, p)]
        def anchor(p):
            for q in sorted(nodes, key=lambda q: math.dist(p, q)):
                if self.segment_free(p, q):
                    return q
            raise ValueError('No map connection')
        source, target = anchor(start), anchor(goal)
        lookup = {(round((x-xmin)/resolution), round((y-ymin)/resolution)): (x, y) for x, y in nodes}
        queue, costs, parents = [(0., source)], {source: 0.}, {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == target:
                path = [goal, target]
                while current != source:
                    current = parents[current]
                    path.append(current)
                path.append(start)
                return list(reversed(path))
            ix, iy = round((current[0]-xmin)/resolution), round((current[1]-ymin)/resolution)
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt = lookup.get((ix+dx, iy+dy))
                if nxt is None or not self.segment_free(current, nxt):
                    continue
                cost = costs[current] + math.dist(current, nxt)
                if cost < costs.get(nxt, float('inf')):
                    costs[nxt], parents[nxt] = cost, current
                    heapq.heappush(queue, (cost + math.dist(nxt, target), nxt))
        raise ValueError('No collision-free route')

    def populate(self, spec, mj):
        for box in self.boxes:
            spec.worldbody.add_geom(name='scene_' + box.name, type=mj.mjtGeom.mjGEOM_BOX,
                                    pos=[box.x, box.y, box.height/2],
                                    size=[box.hx, box.hy, box.height/2], rgba=box.color,
                                    friction=[.8, .01, .001])
        spec.worldbody.add_site(name='navigation_goal', type=mj.mjtGeom.mjGEOM_CYLINDER,
                                pos=[4., 0., .012], size=[.15, .012, .012], rgba=[.1,.8,.2,.7])
        spec.worldbody.add_light(pos=[2., 0., 4.], dir=[0.,0.,-1.])
