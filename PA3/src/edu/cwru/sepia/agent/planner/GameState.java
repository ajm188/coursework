package edu.cwru.sepia.agent.planner;

import edu.cwru.sepia.agent.planner.actions.DepositGold;
import edu.cwru.sepia.agent.planner.actions.DepositWood;
import edu.cwru.sepia.agent.planner.actions.HarvestGold;
import edu.cwru.sepia.agent.planner.actions.HarvestWood;
import edu.cwru.sepia.agent.planner.actions.StripsAction;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;

import java.util.ArrayList;
import java.util.List;

/**
 * This class is used to represent the state of the game after applying one of the avaiable actions. It will also
 * track the A* specific information such as the parent pointer and the cost and heuristic function. Remember that
 * unlike the path planning A* from the first assignment the cost of an action may be more than 1. Specifically the cost
 * of executing a compound action such as move can be more than 1. You will need to account for this in your heuristic
 * and your cost function.
 *
 * The first instance is constructed from the StateView object (like in PA2). Implement the methods provided and
 * add any other methods and member variables you need.
 *
 * Some useful API calls for the state view are
 *
 * state.getXExtent() and state.getYExtent() to get the map size
 *
 * I recommend storing the actions that generated the instance of the GameState in this class using whatever
 * class/structure you use to represent actions.
 */
public class GameState implements Comparable<GameState> {

	private State.StateView stateView;
	private int playernum;
	private int requiredGold;
	private int requiredWood;
	private boolean buildPeasants;
	private GameState parent;

	private Unit.UnitView peasantView;
	private Unit.UnitView townHallView;
	
	private List<ResourceNode.ResourceView> resourceViews;

	private StripsAction stripsAction;

    /**
     * Construct a GameState from a stateview object. This is used to construct the initial search node. All other
     * nodes should be constructed from the another constructor you create or by factory functions that you create.
     *
     * @param state The current stateview at the time the plan is being created
     * @param playernum The player number of agent that is planning
     * @param requiredGold The goal amount of gold (e.g. 200 for the small scenario)
     * @param requiredWood The goal amount of wood (e.g. 200 for the small scenario)
     * @param buildPeasants True if the BuildPeasant action should be considered
     */
    public GameState(State.StateView state, int playernum, int requiredGold, int requiredWood, boolean buildPeasants) {
    	construct(state, playernum, requiredGold, requiredWood, buildPeasants, null, null);
    }

    public GameState(State.StateView state, int playernum, int requiredGold, int requiredWood, boolean buildPeasants, GameState parent, StripsAction action) {
    	construct(state, playernum, requiredGold, requiredWood, buildPeasants, parent, action);
    }

    private void construct(State.StateView state, int playernum, int requiredGold, int requiredWood, boolean buildPeasants, GameState parent, StripsAction action) {
    	this.stateView = state;
        this.playernum = playernum;
        this.requiredGold = requiredGold;
        this.requiredWood = requiredWood;
        this.buildPeasants = buildPeasants;
        this.parent = parent;
        this.stripsAction = action;
        
        extractInfoFromStateView(state);
    }

    private void extractInfoFromStateView(State.StateView state) {
    	for (Unit.UnitView unitView : state.getUnits(playernum)) {
    		if (unitView.getTemplateView().getName().equals("peasant")) {
    			peasantView = unitView;
    		} else if (unitView.getTemplateView().getName().equals("townhall")) {
    			townHallView = unitView;
    		}
    	}
    	this.resourceViews = state.getAllResourceNodes();
    }

    /**
     * Unlike in the first A* assignment there are many possible goal states. As long as the wood and gold requirements
     * are met the peasants can be at any location and the capacities of the resource locations can be anything. Use
     * this function to check if the goal conditions are met and return true if they are.
     *
     * @return true if the goal conditions are met in this instance of game state.
     */
    public boolean isGoal() {
        return this.stateView.getResourceAmount(playernum, ResourceType.GOLD) >= this.requiredGold &&
        		this.stateView.getResourceAmount(playernum, ResourceType.WOOD) >= this.requiredWood;
    }

    /**
     * The branching factor of this search graph are much higher than the planning. Generate all of the possible
     * successor states and their associated actions in this method.
     *
     * @return A list of the possible successor states and their associated actions
     */
    public List<GameState> generateChildren() {
    	List<GameState> children = new ArrayList<GameState>();
    	
        Position peasantPosition = new Position(peasantView.getXPosition(), peasantView.getYPosition());
        Position townHallPosition = new Position(townHallView.getXPosition(), townHallView.getYPosition());
        
        List<Position> goldPositions = new ArrayList<Position>();
        List<Position> treePositions = new ArrayList<Position>();
        for (ResourceNode.ResourceView resourceView : resourceViews) {
        	Position p = new Position(resourceView.getXPosition(), resourceView.getYPosition());
        	if (resourceView.getType() == ResourceNode.Type.GOLD_MINE) {
        		goldPositions.add(p);
        	} else if (resourceView.getType() == ResourceNode.Type.TREE) {
        		treePositions.add(p);
        	}
        }
        
        // try deposits
        DepositGold depositGoldAction = new DepositGold(peasantPosition, townHallPosition);
        if (depositGoldAction.preconditionsMet(this)) {
        	children.add(depositGoldAction.apply(this));
        }
        DepositWood depositWoodAction = new DepositWood(peasantPosition, townHallPosition);
        if (depositWoodAction.preconditionsMet(this)) {
        	children.add(depositWoodAction.apply(this));
        }
        
        // try harvesting
        for (Position goldMinePosition : goldPositions) {
        	HarvestGold harvest = new HarvestGold(peasantPosition, goldMinePosition);
        	if (harvest.preconditionsMet(this)) {
        		children.add(harvest.apply(this));
        	}
        }
        for (Position treePosition : treePositions) {
        	HarvestWood harvest = new HarvestWood(peasantPosition, treePosition);
        	if (harvest.preconditionsMet(this)) {
        		children.add(harvest.apply(this));
        	}
        }
        
        return children;
    }

    /**
     * Write your heuristic function here. Remember this must be admissible for the properties of A* to hold. If you
     * can come up with an easy way of computing a consistent heuristic that is even better, but not strictly necessary.
     *
     * Add a description here in your submission explaining your heuristic.
     *
     * @return The value estimated remaining cost to reach a goal state from this state.
     */
    public double heuristic() {
        // TODO: Implement me!
        return 0.0;
    }

    /**
     *
     * Write the function that computes the current cost to get to this node. This is combined with your heuristic to
     * determine which actions/states are better to explore.
     *
     * @return The current cost to reach this goal
     */
    public double getCost() {
        // TODO: Implement me!
        return 0.0;
    }
	
	public State.StateView getStateView() {
		return this.stateView;
	}
	
	public int getPlayerNum() {
		return this.playernum;
	}
	
	public int getRequiredGold() {
		return this.requiredGold;
	}
	
	public int getRequiredWood() {
		return this.requiredWood;
	}
	
	public boolean getBuildPeasants() {
		return this.buildPeasants;
	}
	
	public GameState getParent() {
		return this.parent;
	}
	
	public Unit.UnitView getPeasantView() {
		return this.peasantView;
	}

	public Unit.UnitView getTownHallView() {
		return this.townHallView;
	}

	public StripsAction getStripsAction() {
		return this.stripsAction;
	}

    /**
     * This is necessary to use your state in the Java priority queue. See the official priority queue and Comparable
     * interface documentation to learn how this function should work.
     *
     * @param o The other game state to compare
     * @return 1 if this state costs more than the other, 0 if equal, -1 otherwise
     */
    @Override
    public int compareTo(GameState o) {
        // TODO: Implement me!
        return 0;
    }

    /**
     * This will be necessary to use the GameState as a key in a Set or Map.
     *
     * @param o The game state to compare
     * @return True if this state equals the other state, false otherwise.
     */
    @Override
    public boolean equals(Object o) {
        // TODO: Implement me!
        return false;
    }

    /**
     * This is necessary to use the GameState as a key in a HashSet or HashMap. Remember that if two objects are
     * equal they should hash to the same value.
     *
     * @return An integer hashcode that is equal for equal states.
     */
    @Override
    public int hashCode() {
        // TODO: Implement me!
        return 0;
    }
}
