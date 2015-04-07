package edu.cwru.sepia.agent.planner;

import edu.cwru.sepia.agent.planner.actions.BuildPeasant;
import edu.cwru.sepia.agent.planner.actions.DepositGold;
import edu.cwru.sepia.agent.planner.actions.DepositWood;
import edu.cwru.sepia.agent.planner.actions.HarvestGold;
import edu.cwru.sepia.agent.planner.actions.HarvestWood;
import edu.cwru.sepia.agent.planner.actions.StripsAction;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.ResourceType;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.ResourceNode.ResourceView;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
	public static class Resource {
		private int id;
		private ResourceNode.Type type;
		private Position position;
		private int amountRemaining;
		
		public Resource(int id, ResourceNode.Type type, Position position, int amountRemaining) {
			this.id = id;
			this.type = type;
			this.position = position;
			this.amountRemaining = amountRemaining;
		}
		
		public int getID() {
			return this.id;
		}
		
		public ResourceNode.Type getType() {
			return this.type;
		}
		
		public Position getPosition() {
			return this.position;
		}
		
		public int getAmountRemaining() {
			return this.amountRemaining;
		}
		
		public void harvest(int quantity) {
			assert this.amountRemaining >= quantity;
			
			this.amountRemaining -= quantity;
		}
		
		public Resource clone() {
			// don't need to deep clone the position, because resources don't move
			return new Resource(id, type, position, amountRemaining);
		}
	}
	
	public static class Peasant {
		private int id;
		private Position position;
		private int cargoAmount;
		private ResourceType cargoType;
		
		public Peasant(int id, Position position) {
			constructor(id, position, 0, null);
		}
		
		public Peasant(int id, Position position, int cargoAmount, ResourceType cargoType) {
			constructor(id, position, cargoAmount, cargoType);
		}
		
		private void constructor(int id, Position position, int cargoAmount, ResourceType cargoType) {
			this.id = id;
			this.position = position;
			this.cargoAmount = cargoAmount;
			this.cargoType = cargoType;
		}
		
		public int getID() {
			return this.id;
		}
		
		public Position getPosition() {
			return this.position;
		}
		
		public void move(Position position) {
			this.position = position;
		}
		
		public int getCargoAmount() {
			return this.cargoAmount;
		}
		
		public ResourceType getCargoType() {
			return this.cargoType;
		}
		
		public void harvest(int amount, ResourceType type) {
			assert cargoAmount == 0;
			assert cargoType == null;
			
			this.cargoAmount = amount;
			this.cargoType = type;
		}
		
		public void deposit() {
			this.cargoAmount = 0;
			this.cargoType = null;
		}
	
		public Peasant clone() {
			return new Peasant(id, new Position(position.x, position.y), cargoAmount, cargoType);
		}
	}
	
	public static class TownHall {
		private int id;
		private Position position;
		
		public TownHall(int id, Position position) {
			this.id = id;
			this.position = position;
		}
		
		public int getID() {
			return this.id;
		}
		
		public Position getPosition() {
			return this.position;
		}
		
		public TownHall clone() {
			// town hall's don't move, so positions don't need to be cloned
			return new TownHall(id, position);
		}
	}

	private int playernum;
	private int requiredGold;
	private int requiredWood;
	private boolean buildPeasants;
	private GameState parent;
	private double cost;
	
	private TownHall townHall;
	private Map<Integer, Peasant> peasants;
	private Map<Integer,Resource> resources;

	private StripsAction stripsAction;
	
	private int goldTotal;
	private int woodTotal;

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
    	construct(state, playernum, requiredGold, requiredWood, buildPeasants, 0.0, null, null);
    }
    
    public GameState(GameState parent, StripsAction action) {
    	this.parent = parent;
    	this.playernum = parent.playernum;
    	this.requiredGold = parent.requiredGold;
    	this.requiredWood = parent.requiredWood;
    	this.buildPeasants = parent.buildPeasants;
    	this.goldTotal = parent.goldTotal;
    	this.woodTotal = parent.woodTotal;
    	
    	this.stripsAction = action;
    	
    	// clone the resources
    	this.resources = new HashMap<Integer,Resource>();
    	for (Resource resource : parent.resources.values()) {
    		this.resources.put(resource.id,resource.clone());
    	}
    	// clone the peasant(s) and townhall
    	this.peasants = new HashMap<Integer, Peasant>();
    	for (Peasant peasant : parent.peasants.values()) {
    		this.peasants.put(peasant.getID(), peasant.clone());
    	}
    	this.townHall = parent.townHall.clone();
    	
    	this.cost = parent.cost + action.getCost();
    }

    private void construct(State.StateView state, int playernum, int requiredGold, int requiredWood, boolean buildPeasants, double cost, GameState parent, StripsAction action) {
        this.playernum = playernum;
        this.requiredGold = requiredGold;
        this.requiredWood = requiredWood;
        this.buildPeasants = buildPeasants;
        this.parent = parent;
        this.stripsAction = action;
        
        this.cost = cost;
        if (parent != null) {
        	this.cost += parent.getCost();
        }
        
        extractInfoFromStateView(state);
    }

    private void extractInfoFromStateView(State.StateView state) {
    	this.peasants = new HashMap<Integer, Peasant>();
    	for (Unit.UnitView unitView : state.getUnits(playernum)) {
    		if (unitView.getTemplateView().getName().equals("Peasant")) {
    			Position peasantPosition = new Position(unitView.getXPosition(), unitView.getYPosition());
    			// currently assuming the peasant always starts carrying nothing
    			Peasant peasant = new Peasant(unitView.getID(), peasantPosition);
    			this.peasants.put(peasant.id, peasant);
    		} else if (unitView.getTemplateView().getName().equals("TownHall")) {
    			Position townHallPosition = new Position(unitView.getXPosition(), unitView.getYPosition());
    			townHall = new TownHall(unitView.getID(), townHallPosition);
    		}
    	}
    	this.resources = new HashMap<Integer,Resource>();
    	for (ResourceView resourceView : state.getAllResourceNodes()) {
    		Position resourcePosition = new Position(resourceView.getXPosition(), resourceView.getYPosition());
    		this.resources.put(resourceView.getID(),new Resource(resourceView.getID(),
    										resourceView.getType(),
    										resourcePosition, 
    										resourceView.getAmountRemaining()));
    	}
    }

    /**
     * Unlike in the first A* assignment there are many possible goal states. As long as the wood and gold requirements
     * are met the peasants can be at any location and the capacities of the resource locations can be anything. Use
     * this function to check if the goal conditions are met and return true if they are.
     *
     * @return true if the goal conditions are met in this instance of game state.
     */
    public boolean isGoal() {
        return this.goldTotal >= this.requiredGold && this.woodTotal >= this.requiredWood;
    }

    /**
     * The branching factor of this search graph are much higher than the planning. Generate all of the possible
     * successor states and their associated actions in this method.
     *
     * @return A list of the possible successor states and their associated actions
     */
    public List<GameState> generateChildren() {
    	List<GameState> children = new ArrayList<GameState>();
    	        
        List<Resource> goldMines = new ArrayList<Resource>();
        List<Resource> trees = new ArrayList<Resource>();
        for (Resource resource : resources.values()) {
        	if (resource.getType() == ResourceNode.Type.GOLD_MINE) {
        		goldMines.add(resource);
        	} else if (resource.getType() == ResourceNode.Type.TREE) {
        		trees.add(resource);
        	}
        }
        
        for (Peasant peasant : this.peasants.values()) {
        	// try deposits
	        DepositGold depositGoldAction = new DepositGold(peasant, townHall);
	        if (depositGoldAction.preconditionsMet(this)) {
	        	children.add(depositGoldAction.apply(this));
	        }
	        DepositWood depositWoodAction = new DepositWood(peasant, townHall);
	        if (depositWoodAction.preconditionsMet(this)) {
	        	children.add(depositWoodAction.apply(this));
	        }
        
	        // try harvesting
	        for (Resource goldMine : goldMines) {
	        	HarvestGold harvest = new HarvestGold(peasant, goldMine);
	        	if (harvest.preconditionsMet(this)) {
	        		children.add(harvest.apply(this));
	        	}
	        }
	        for (Resource tree : trees) {
	        	HarvestWood harvest = new HarvestWood(peasant, tree);
	        	if (harvest.preconditionsMet(this)) {
	        		children.add(harvest.apply(this));
	        	}
	        }
        }
        
        if (buildPeasants && this.peasants.size() < 3) {
        	// try building peasants
        	BuildPeasant build = new BuildPeasant();
        	if (build.preconditionsMet(this)) {
        		children.add(build.apply(this));
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
    	int goldToGo = this.requiredGold - this.goldTotal;
    	if (goldToGo < 0) {
    		goldToGo = 0;
    	}
    	
    	int woodToGo = this.requiredWood - this.woodTotal;
    	if (woodToGo < 0) {
    		woodToGo = 0;
    	}
        
    	/*
    	 *  having more peasants is way better
    	 *  multiplying the numerator produces a larger spread in the
    	 *  values for the heuristics. then, dividing that by the number of
    	 *  peasants to a power, causes the 3-peasant states to have a heuristic
    	 *  way closer to 0 (which is good) than the 2-peasant states, which are
    	 *  way closer to 0 than the 1-peasant states.
    	 */
    	return 100 * (goldToGo + woodToGo) / (Math.pow(this.peasants.size(), 3));
    }

    /**
     *
     * Write the function that computes the current cost to get to this node. This is combined with your heuristic to
     * determine which actions/states are better to explore.
     *
     * @return The current cost to reach this goal
     */
    public double getCost() {
        return this.cost + heuristic();
    }
    
    /**
     * Create a new peasant, and place him/her next to the town hall, carrying nothing.
     */
    public void addPeasant() {
    	// assume there is an open square
    	Position newPeasantPosition = this.townHall.getPosition().getAdjacentPositions().get(0);
    	
    	Peasant recruit = new Peasant(this.peasants.size() + 1, newPeasantPosition);
    	this.peasants.put(recruit.id, recruit);
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
	
	public Map<Integer, Peasant> getPeasants() {
		return this.peasants;
	}
	
	public TownHall getTownHall() {
		return this.townHall;
	}
	
	public Map<Integer,Resource> getResources(){
		return this.resources;
	}
	
	public int getGoldTotal() {
		return this.goldTotal;
	}
	
	public void addGold(int amount) {
		this.goldTotal += amount;
	}
	
	public void addWood(int amount) {
		this.woodTotal += amount;
	}
	
	public GameState getParent() {
		return this.parent;
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
        return (int) (this.getCost() - o.getCost());
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
    	return 0; // TODO
    }
}
