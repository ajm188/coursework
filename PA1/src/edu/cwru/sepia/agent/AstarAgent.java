package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.AStarNode;
import edu.cwru.sepia.util.Direction;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;

public class AstarAgent extends Agent {

    class MapLocation {
        public int x, y;

        public MapLocation(int x, int y, MapLocation cameFrom, float cost) {
            this.x = x;
            this.y = y;
        }
    }

    Stack<MapLocation> path;
    int footmanID, townhallID, enemyFootmanID;
    MapLocation nextLoc;

    private long totalPlanTime = 0; // nsecs
    private long totalExecutionTime = 0; //nsecs

    public AstarAgent(int playernum) {
        super(playernum);
	// Declare Closed List as a set
        // Declare Open List (Frontier) as a Priority Queue
	// start with root node (initial location of agent)
	// while the open set is not empty
		// pop a node off the open list
		// if it's the goal, you're done
		// else "Search Algorithm Junk"
			// compute heuristic 
			// explore nodes
	//Return No Available Path if no solution found
        System.out.println("Constructed AstarAgent");
    }

    private void setFootmanID(State.StateView newstate) throws Exception {
        List<Integer> unitIDs = newstate.getUnitIds(playernum);
        // Make sure we have some units
        if(unitIDs.size() == 0) {
            throw new Exception("No units found!");
        }

        footmanID = unitIDs.get(0);
        // Make sure it's actually a footman
        if(!newstate.getUnit(footmanID).getTemplateView().getName().equals("Footman")) {
            throw new Exception("Footman unit not found");
        }
    }

    private int findEnemyPlayerNum(State.StateView newstate) throws Exception {
        Integer[] playerNums = newstate.getPlayerNumbers();
        int enemyPlayerNum = -1;
        for (Integer playerNum : playerNums) {
            if (playerNum != playernum) {
                enemyPlayerNum = playerNum;
                break;
            }
        }
        if (enemyPlayerNum == -1) {
            throw new Exception("Failed to get enemy playernumber");
        }
        return enemyPlayerNum;
    }

    private List<Integer> getEnemyUnitIDs(State.StateView newstate, int enemyPlayerNum) throws Exception {
        List<Integer> enemyUnitIDs = newstate.getUnitIds(enemyPlayerNum);
        if (enemyUnitIDs.size() == 0) {
            throw new Exception("Failed to find enemy units");
        }
        return enemyUnitIDs;
    }

    private void setTownhallID(State.StateView newstate, int enemyPlayerNum) throws Exception {
        List<Integer> enemyUnitIDs = getEnemyUnitIDs(newstate, enemyPlayerNum);

        townhallID = -1;
        enemyFootmanID = -1;

        for (Integer unitID : enemyUnitIDs) {
            String unitType = newstate.getUnit(unitID).getTemplateView().getName().toLowerCase();
            if (unitType.equals("townhall")) {
                townhallID = unitID;    
            } else if (unitType.equals("footman")) {
                enemyFootmanID = unitID;
            } else {
                System.err.println("Unknown unit type");
            }
        }

        if (townhallID == -1) {
            throw new Exception("Error: Couldn't find townhall");
        }
    }

    @Override
    public Map<Integer, Action> initialStep(State.StateView newstate, History.HistoryView statehistory) {
        try {
            setFootmanID(newstate); // find the footman location
            setTownhallID(newstate, findEnemyPlayerNum(newstate));
        } catch (Exception e) {
            System.err.println(e.getMessage());
            return null;
        }
        
        long startTime = System.nanoTime();
        path = findPath(newstate);
        totalPlanTime += System.nanoTime() - startTime;

        return middleStep(newstate, statehistory);
    }

    /**
     * Replan the path and return the time taken to replan the path
     */
    private long replanPath(State.StateView newstate) {
        long planStartTime, planTime;
        planStartTime = System.nanoTime();

        // actually replan the path. the rest is just time auditing stuff
        path = findPath(newstate);
        
        planTime = System.nanoTime() - planStartTime;
        totalPlanTime += planTime;
        return planTime;
    }

    /**
     * Returns whether the given unit is at nextLoc.
     * nextLoc is the instance variable stored in the AstarAgent class
     * @param unit
     */
    private boolean isUnitAtNextLoc(Unit.UnitView unit) {
        return unit.getXPosition() == nextLoc.x && unit.getYPosition() == nextLoc.y;
    }

    /**
     * Determines whether two units are neighbors on the map (if they are within one tile in any direction)
     * @param unitA @param unitB The two units to be compared
     */
    private boolean areNeighbors(Unit.UnitView unitA, Unit.UnitView unitB) {
        return Math.abs(unitA.getXPosition() - unitB.getXPosition()) <= 1 &&
            Math.abs(unitA.getYPosition() - unitB.getYPosition()) <= 1;
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView newstate, History.HistoryView statehistory) {
        long startTime = System.nanoTime();
        long planTime = 0;

        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        if (shouldReplanPath(newstate, statehistory, path)) {
            planTime = replanPath(newstate);
        }

        Unit.UnitView footmanUnit = newstate.getUnit(footmanID);

        if (!path.empty() && (nextLoc == null || isUnitAtNextLoc(footmanUnit))) {
            // stat moving to the next step in the path
            nextLoc = path.pop();
            System.out.println("Moving to (" + nextLoc.x + ", " + nextLoc.y + ")");
        }

        if (nextLoc != null && !isUnitAtNextLoc(footmanUnit)) {
            // figure out the next direction and then put the action into the map
            actions.put(footmanID, Action.createPrimitiveMove(footmanID, getNextDirection(footmanUnit)));
        } else {
            Unit.UnitView townhallUnit = newstate.getUnit(townhallID);

            // if townhall was destroyed on the last turn
            if (townhallUnit == null) {
                terminalStep(newstate, statehistory);
                return actions;
            }
            if (!areNeighbors(footmanUnit, townhallUnit)) {
                System.err.println("Invalid plan. Cannot attack townhall");
                totalExecutionTime += System.nanoTime() - startTime - planTime;
                return actions;
            } else {
                System.out.println("Attacking TownHall");
                // if no more movements in the planned path then attack
                actions.put(footmanID, Action.createPrimitiveAttack(footmanID, townhallID));
            }
        }

        totalExecutionTime += System.nanoTime() - startTime - planTime;
        return actions;
    }

    @Override
    public void terminalStep(State.StateView newstate, History.HistoryView statehistory) {
        System.out.println("Total turns: " + newstate.getTurnNumber());
        System.out.println("Total planning time: " + totalPlanTime/1e9);
        System.out.println("Total execution time: " + totalExecutionTime/1e9);
        System.out.println("Total time: " + (totalExecutionTime + totalPlanTime)/1e9);
    }

    @Override
    public void savePlayerData(OutputStream os) {

    }

    @Override
    public void loadPlayerData(InputStream is) {

    }

    /**
     * You will implement this method.
     *
     * This method should return true when the path needs to be replanned
     * and false otherwise. This will be necessary on the dynamic map where the
     * footman will move to block your unit.
     *
     * @param state
     * @param history
     * @param currentPath
     * @return
     */
    private boolean shouldReplanPath(State.StateView state, History.HistoryView history, Stack<MapLocation> currentPath)
    {
        return false;
    }

    /**
     * This method is implemented for you. You should look at it to see examples of
     * how to find units and resources in Sepia.
     *
     * @param state
     * @return
     */
    private Stack<MapLocation> findPath(State.StateView state)
    {
        Unit.UnitView townhallUnit = state.getUnit(townhallID);
        Unit.UnitView footmanUnit = state.getUnit(footmanID);

        MapLocation startLoc = new MapLocation(footmanUnit.getXPosition(), footmanUnit.getYPosition(), null, 0);

        MapLocation goalLoc = new MapLocation(townhallUnit.getXPosition(), townhallUnit.getYPosition(), null, 0);

        MapLocation footmanLoc = null;
        if(enemyFootmanID != -1) {
            Unit.UnitView enemyFootmanUnit = state.getUnit(enemyFootmanID);
            footmanLoc = new MapLocation(enemyFootmanUnit.getXPosition(), enemyFootmanUnit.getYPosition(), null, 0);
        }

        // get resource locations
        List<Integer> resourceIDs = state.getAllResourceIds();
        Set<MapLocation> resourceLocations = new HashSet<MapLocation>();
        for(Integer resourceID : resourceIDs)
        {
            ResourceNode.ResourceView resource = state.getResourceNode(resourceID);

            resourceLocations.add(new MapLocation(resource.getXPosition(), resource.getYPosition(), null, 0));
        }

        return AstarSearch(startLoc, goalLoc, state.getXExtent(), state.getYExtent(), footmanLoc, resourceLocations);
    }
    /**
     * This is the method you will implement for the assignment. Your implementation
     * will use the A* algorithm to compute the optimum path from the start position to
     * a position adjacent to the goal position.
     *
     * You will return a Stack of positions with the top of the stack being the first space to move to
     * and the bottom of the stack being the last space to move to. If there is no path to the townhall
     * then return null from the method and the agent will print a message and do nothing.
     * The code to execute the plan is provided for you in the middleStep method.
     *
     * As an example consider the following simple map
     *
     * F - - - -
     * x x x - x
     * H - - - -
     *
     * F is the footman
     * H is the townhall
     * x's are occupied spaces
     *
     * xExtent would be 5 for this map with valid X coordinates in the range of [0, 4]
     * x=0 is the left most column and x=4 is the right most column
     *
     * yExtent would be 3 for this map with valid Y coordinates in the range of [0, 2]
     * y=0 is the top most row and y=2 is the bottom most row
     *
     * resourceLocations would be {(0,1), (1,1), (2,1), (4,1)}
     *
     * The path would be
     *
     * (1,0)
     * (2,0)
     * (3,1)
     * (2,2)
     * (1,2)
     *
     * Notice how the initial footman position and the townhall position are not included in the path stack
     *
     * @param start Starting position of the footman
     * @param goal MapLocation of the townhall
     * @param xExtent Width of the map
     * @param yExtent Height of the map
     * @param resourceLocations Set of positions occupied by resources
     * @return Stack of positions with top of stack being first move in plan
     */
    private Stack<MapLocation> AstarSearch(MapLocation start, MapLocation goal, int xExtent, int yExtent, MapLocation enemyFootmanLoc, Set<MapLocation> resourceLocations)
    {
	// Declare Closed List as a set
        Set<AStarNode> closedList = new HashSet<AStarNode>();
	// Declare Open List (Frontier) as a Priority Queue
	PriorityQueue<AStarNode> openList = new PriorityQueue<AStarNode>();
	// start with root node (initial location of agent)
	AStarNode root = new AStarNode(start.x, start.y, chebyshev(start, goal));
	AStarNode aStarGoal = new AStarNode(goal.x, goal.y, 0);
	// while the open set is not empty
	while (openList.isEmpty()){
		// pop a node off the open list
		AStarNode node = openList.poll();
		// if it's the goal, you're done
		if (root.equals(aStarGoal){
			return null;
		}
		// else "Search Algorithm Junk"
			// compute heuristic
	} 
	return new Stack<MapLocation>();
    }

    private int chebyshev(MapLocation loc1, MapLocation loc2){
	return Math.max(Math.abs(loc2.x-loc1.x), Math.abs(loc2.y-loc1.y));
    }    

    /**
     * Overload of the original getNextDirection method.
     * Determines the direction @param unit should move
     * relative to nextLoc
     */
    private Direction getNextDirection(Unit.UnitView unit) {
        return getNextDirection(nextLoc.x - unit.getXPosition(), nextLoc.y - unit.getYPosition());
    }

    /**
     * Primitive actions take a direction (e.g. NORTH, NORTHEAST, etc)
     * This converts the difference between the current position and the
     * desired position to a direction.
     *
     * @param xDiff Integer equal to 1, 0 or -1
     * @param yDiff Integer equal to 1, 0 or -1
     * @return A Direction instance (e.g. SOUTHWEST) or null in the case of error
     */
    private Direction getNextDirection(int xDiff, int yDiff) {

        // figure out the direction the footman needs to move in
        if(xDiff == 1 && yDiff == 1)
        {
            return Direction.SOUTHEAST;
        }
        else if(xDiff == 1 && yDiff == 0)
        {
            return Direction.EAST;
        }
        else if(xDiff == 1 && yDiff == -1)
        {
            return Direction.NORTHEAST;
        }
        else if(xDiff == 0 && yDiff == 1)
        {
            return Direction.SOUTH;
        }
        else if(xDiff == 0 && yDiff == -1)
        {
            return Direction.NORTH;
        }
        else if(xDiff == -1 && yDiff == 1)
        {
            return Direction.SOUTHWEST;
        }
        else if(xDiff == -1 && yDiff == 0)
        {
            return Direction.WEST;
        }
        else if(xDiff == -1 && yDiff == -1)
        {
            return Direction.NORTHWEST;
        }

        System.err.println("Invalid path. Could not determine direction");
        return null;
    }
}
