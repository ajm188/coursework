package edu.cwru.sepia.environment.model;

import edu.cwru.sepia.util.Direction;

/**
 * For assignment 1, we need some of the variables of A* nodes exposed.
 * So, we subclassed it to expose them here.
 */
public class ExposedAStarNode extends AStarNode {
    public ExposedAStarNode(int x, int y, int g, int value, ExposedAStarNode previous, Direction directionfromprevious) {
        super(x, y, g, value, previous, directionfromprevious);
    }

    public ExposedAStarNode(int x, int y, int g, int value, ExposedAStarNode previous, Direction directionfromprevious, int durativesteps) {
        super(x, y, g, value, previous, directionfromprevious, durativesteps);
    }

    public ExposedAStarNode(int x, int y, int distfromgoal) {
        super(x, y, distfromgoal);
        assert previous == null; // make extra sure that previous gets set to null
    }

    public int x() {
        return this.x;
    }

    public int y() {
        return this.y;
    }

    public int g() {
        return this.g;
    }

    public int value() {
        return this.value;
    }

    public ExposedAStarNode previous() {
        return (ExposedAStarNode) this.previous;
    }
}

