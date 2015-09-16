package edu.cwru.sepia.agent.minimax;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.DirectedAction;
import edu.cwru.sepia.agent.Agent;
import edu.cwru.sepia.agent.AstarAgent;
import edu.cwru.sepia.agent.AstarAgent.MapLocation;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.Direction;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.Stack;

public class MinimaxAlphaBeta extends Agent {

    private final int numPlys;
    
    private Map<Integer, Stack<MapLocation>> astarPaths = new HashMap<Integer, Stack<MapLocation>>();
    private Set<MapLocation> resourceLocations = null;

    public MinimaxAlphaBeta(int playernum, String[] args)
    {
        super(playernum);

        if(args.length < 1)
        {
            System.err.println("You must specify the number of plys");
            System.exit(1);
        }

        numPlys = Integer.parseInt(args[0]);
        System.out.println(numPlys);
    }

    @Override
    public Map<Integer, Action> initialStep(State.StateView newstate, History.HistoryView statehistory) {
        return middleStep(newstate, statehistory);
    }

    @Override
    public Map<Integer, Action> middleStep(State.StateView newstate, History.HistoryView statehistory) {
    	if (resourceLocations == null)
    	{
    		// these will be used in the A* search
    		// they will also never change, so this code should only be executed once in a full run of the agent
    		resourceLocations = new HashSet<MapLocation>();
    		for (ResourceNode.ResourceView resource : newstate.getAllResourceNodes())
    		{
    			resourceLocations.add(new MapLocation(resource.getXPosition(), resource.getYPosition(), null, 0));
    		}
    	}
    	
    	List<UnitView> footmen = newstate.getUnits(0);
    	List<UnitView> archers = newstate.getUnits(1);
    	
    	Set<MapLocation> archerLocs = new HashSet<MapLocation>();
    	for (UnitView archer : archers)
    	{
    		archerLocs.add(new MapLocation(archer.getXPosition(), archer.getYPosition(), null, 0));
    	}
    	
    	for (UnitView footman : footmen)
    	{
    		Stack<MapLocation> path = astarPaths.get(footman.getID());
    		if (path == null || path.isEmpty())
    		{
    			astarPaths.put(footman.getID(),
    							AstarAgent.AstarSearch(new MapLocation(footman.getXPosition(), footman.getYPosition(), null, 0),
    													archerLocs, newstate.getXExtent(), newstate.getYExtent(), resourceLocations));
    		}
    	}

        boolean isMax = true;
        GameStateChild bestChild = alphaBetaSearch(new GameStateChild(newstate,isMax),
                numPlys,
                Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY);
        
        for (Integer footmanID : bestChild.action.keySet())
        {
        	// Should we recalculate anything?
        	Action action = bestChild.action.get(footmanID);
        	if (action.getType() == ActionType.PRIMITIVEATTACK)
        	{
        		// definitely want to recalculate, but on the next iteration (in case the archer moves away)
        		astarPaths.put(footmanID, null);
        	}
        	else if (action.getType() == ActionType.PRIMITIVEMOVE)
        	{
        		Direction d = ((DirectedAction) action).getDirection();
        		UnitView footman = newstate.getUnit(footmanID);
        		MapLocation nextLoc = new MapLocation(footman.getXPosition() + d.xComponent(), footman.getYPosition() + d.yComponent(), null, 0);
        		if (nextLoc.equals(astarPaths.get(footmanID).peek()))
        		{
        			// all good! pop the move off the stack
        			astarPaths.get(footmanID).pop();
        		}
        		else
        		{
        			// need to recalculate on the next iteration
        			astarPaths.put(footmanID, null);
        		}
        	}
        }

        return bestChild.action;
    }

    @Override
    public void terminalStep(State.StateView newstate, History.HistoryView statehistory) {

    }

    @Override
    public void savePlayerData(OutputStream os) {

    }

    @Override
    public void loadPlayerData(InputStream is) {

    }

    /**
     * You will implement this.
     *
     * This is the main entry point to the alpha beta search. Refer to the slides, assignment description
     * and book for more information.
     *
     * Try to keep the logic in this function as abstract as possible (i.e. move as much SEPIA specific
     * code into other functions and methods)
     *
     * @param node The action and state to search from
     * @param depth The remaining number of plys under this node
     * @param alpha The current best value for the maximizing node from this node to the root
     * @param beta The current best value for the minimizing node from this node to the root
     * @return The best child of this node with updated values
     */
    public GameStateChild alphaBetaSearch(GameStateChild node, int depth, double alpha, double beta)
    {
        if (depth == 0) {
            return node;
        }

        List<GameStateChild> children = orderChildrenWithHeuristics(node.state.getChildren(), depth);
        GameStateChild bestChild = null;
    
        if (depth % 2 == numPlys % 2) {
        	double nodeUtility = Double.NEGATIVE_INFINITY;
            // MAX's move
            for (GameStateChild child : children) {
                GameStateChild temp = alphaBetaSearch(child, depth - 1, alpha, beta);
                // pass A* paths to the get utility function?
                if (temp.state.getUtility(astarPaths) > nodeUtility) {
                	nodeUtility = temp.state.getUtility(astarPaths);
                	node.state.setUtility(nodeUtility);
                	bestChild = temp;
                }

                alpha = Math.max(alpha, nodeUtility);

                if (beta <= alpha) {
                    break;
                }
            }
        } else {
        	double nodeUtility = Double.POSITIVE_INFINITY;
            // MIN's move
            for (GameStateChild child : children) {
                GameStateChild temp = alphaBetaSearch(child, depth - 1, alpha, beta);
                if (temp.state.getUtility(astarPaths) < nodeUtility) {
                	nodeUtility = temp.state.getUtility(astarPaths);
                	node.state.setUtility(nodeUtility);
                	bestChild = temp;
                }

                beta = Math.min(beta, nodeUtility);

                if (beta <= alpha) {
                    break;
                }
            }
        }
        if (depth == numPlys) {
        	node.action = bestChild.action;
        }
        return node;
    }

    /**
     * You will implement this.
     *
     * Given a list of children you will order them according to heuristics you make up.
     * See the assignment description for suggestions on heuristics to use when sorting.
     *
     * Use this function inside of your alphaBetaSearch method.
     *
     * Include a good comment about what your heuristics are and why you chose them.
     *
     * @param children
     * @return The list of children sorted by your heuristic.
     */
    public List<GameStateChild> orderChildrenWithHeuristics(List<GameStateChild> children, int depth)
    {
        GameStateChild[] childrenArray = (GameStateChild[]) children.toArray(new GameStateChild[0]);
        int index;
        for (int j = 0; j < childrenArray.length - 1; j++) {
            index = j;
            for (int i = j + 1; i < childrenArray.length; i++) {
            	if (depth % 2 == numPlys % 2) {
            		// max's turn: sort in decreasing order
	                if (childrenArray[i].state.getUtility(astarPaths) > childrenArray[index].state.getUtility(astarPaths)) {
	                    index = i;
	                }
            	} else {
            		// min's turn: sort in increasing order
	                if (childrenArray[i].state.getUtility(astarPaths) < childrenArray[index].state.getUtility(astarPaths)) {
	                    index = i;
	                }
            	}
            }
            if (index != j) {
                GameStateChild temp = childrenArray[index];
                childrenArray[index] = childrenArray[j];
                childrenArray[j] = temp;
            }
        }

        return Arrays.asList(childrenArray);
    }
}