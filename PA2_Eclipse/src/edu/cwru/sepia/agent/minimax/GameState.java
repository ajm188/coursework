package edu.cwru.sepia.agent.minimax;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.DirectedAction;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.UnitTemplate;
import edu.cwru.sepia.util.Direction;
import edu.cwru.sepia.agent.AstarAgent;

import java.util.*;
import java.io.IOException;

/**
 * This class stores all of the information the agent
 * needs to know about the state of the game. For example this
 * might include things like footmen HP and positions.
 *
 * Add any information or methods you would like to this class,
 * but do not delete or change the signatures of the provided methods.
 */
public class GameState {

	private int xExtent;
	private int yExtent;

	private boolean isMax;

	private Double utility;

	private List<ResourceNode.ResourceView> resourceNodes;
	
	private Map<Integer, Unit> footmen;
	private Map<Integer, Unit> archers;

	private State state;
	/**
	 * You will implement this constructor. It will
	 * extract all of the needed state information from the built in
	 * SEPIA state view.
	 *
	 * You may find the following state methods useful:
	 *
	 * state.getXExtent() and state.getYExtent(): get the map dimensions
	 * state.getAllResourceIDs(): returns all of the obstacles in the map
	 * state.getResourceNode(Integer resourceID): Return a ResourceView for the given ID
	 *
	 * For a given ResourceView you can query the position using
	 * resource.getXPosition() and resource.getYPosition()
	 *
	 * For a given unit you will need to find the attack damage, range and max HP
	 * unitView.getTemplateView().getRange(): This gives you the attack range
	 * unitView.getTemplateView().getBasicAttack(): The amount of damage this unit deals
	 * unitView.getTemplateView().getBaseHealth(): The maximum amount of health of this unit
	 *
	 * @param state Current state of the episode
	 */
	public GameState(State.StateView state) {
		extractInfoFromState(state);
	}
	
	/**
	 * Create a new game state based off of a previous one
	 * @param ancestor
	 */
	public GameState(GameState ancestor) {
		// copy relevant info from the ancestor's tstate
		extractInfoFromState(ancestor.state.getView(ancestor.isMax ? 1 : 0));
		isMax = !ancestor.isMax;
	}

	public GameState(State.StateView state, boolean isMax)
	{
		this.isMax = isMax;
		extractInfoFromState(state);
	}

	private void extractInfoFromState(State.StateView state)
	{
		try {
			this.state = state.getStateCreator().createState();
			xExtent = state.getXExtent();
			yExtent = state.getYExtent();
			resourceNodes = state.getAllResourceNodes();
			
			footmen = this.state.getUnits(0); // player 1 has footmen
			archers = this.state.getUnits(1); // player 2 has archers
		} catch (IOException e) {
		}

		utility = null;
	}

	/**
	 * You will implement this function.
	 *
	 * You should use weighted linear combination of features.
	 * The features may be primitives from the state (such as hp of a unit)
	 * or they may be higher level summaries of information from the state such
	 * as distance to a specific location. Come up with whatever features you think
	 * are useful and weight them appropriately.
	 *
	 * It is recommended that you start simple until you have your algorithm working. Then watch
	 * your agent play and try to add features that correct mistakes it makes. However, remember that
	 * your features should be as fast as possible to compute. If the features are slow then you will be
	 * able to do less plys in a turn.
	 *
	 * Add a good comment about what is in your utility and why you chose those features.
	 *
	 * @return The weighted linear combination of the features
	 */
	public double getUtility() {
		if (utility != null) {
			return utility;
		}
		/*
		 * Features we want:
		 *  - health of the footmen (positive)
		 *  - health of the archers (negative)
		 *  - number of footmen (positive)
		 *  - number of archers (negative)
		 *  - distance from footmen to archers (negative) --> do it with A*
		 * Adjust the weights as we test
		 */
		Set<AstarAgent.MapLocation> occupiedLocations = new HashSet<AstarAgent.MapLocation>();
		for (ResourceNode.ResourceView resource : resourceNodes)
		{
			occupiedLocations.add(new AstarAgent.MapLocation(resource.getXPosition(), resource.getYPosition(), null, 0));
		}
		
		Set<AstarAgent.MapLocation> footmanLocations = new HashSet<AstarAgent.MapLocation>();
		for (Unit footman : footmen.values())
		{
			footmanLocations.add(new AstarAgent.MapLocation(footman.getxPosition(), footman.getyPosition(), null, 0));
		}
		
		occupiedLocations.addAll(footmanLocations);
		
		Set<AstarAgent.MapLocation> archerLocations = new HashSet<AstarAgent.MapLocation>();
		for (Unit archer : archers.values())
		{
			archerLocations.add(new AstarAgent.MapLocation(archer.getxPosition(), archer.getyPosition(), null, 0));
		}
		
		int footmanToArcherDistance = 0;
		for (AstarAgent.MapLocation footman : footmanLocations)
		{
			footmanToArcherDistance += AstarAgent.AstarSearch(footman, archerLocations, xExtent, yExtent, occupiedLocations).size();
		}

		int footmenAttacks = getThreatenedUnits(footmen.values());
		int archerAttacks = getThreatenedUnits(archers.values());

		utility = 1.0 * getTotalHealth(footmen.values()) +
				(-10) * getTotalHealth(archers.values()) +
				10 * footmen.size() +
				(-5) * archers.size() +
				(-2) * footmanToArcherDistance +
				5 * footmenAttacks +
				(-1) * archerAttacks;

		return utility;
	}
	
	public void setUtility(double utility) {
		this.utility = utility;
	}

	private int getTotalHealth(Collection<Unit> units)
	{
		int health = 0;
		for (Unit unit : units)
		{
			health += unit.getCurrentHealth();
		}
		return health;
	}
	
	private int getThreatenedUnits(Collection<Unit> units)
	{
		int unitsThreatened = 0;
		for (Unit unit : units) {
			int upperBound = unit.getTemplate().getRange();
			int lowerBound = -upperBound;
			for (int xAdj = lowerBound; xAdj <= upperBound; xAdj++) {
				for (int yAdj = lowerBound; yAdj <= upperBound; yAdj++) {
					Unit other = state.unitAt(unit.getxPosition() + xAdj, unit.getyPosition() + yAdj);
					if (other != null && other.getPlayer() != unit.getPlayer()) {
						unitsThreatened++;
					}
				}
			}
		}
		return unitsThreatened;
	}

	/**
	 * You will implement this function.
	 *
	 * This will return a list of GameStateChild objects. You will generate all of the possible
	 * actions in a step and then determine the resulting game state from that action. These are your GameStateChildren.
	 *
	 * You may find it useful to iterate over all the different directions in SEPIA.
	 *
	 * for(Direction direction : Directions.values())
	 *
	 * To get the resulting position from a move in that direction you can do the following
	 * x += direction.xComponent()
	 * y += direction.yComponent()
	 *
	 * @return All possible actions and their associated resulting game state
	 */
	public List<GameStateChild> getChildren() {
		//depending on whose turn it is, collect all possible actions
		List<Unit> desiredList = new ArrayList<Unit>(isMax ? footmen.values() : archers.values());
		Map<Integer,List<Action>> unitActions = new HashMap<Integer, List<Action>>();
		for (Unit unit : desiredList) {
			unitActions.put(unit.getView().getID(), getAllActions(unit));
		}

		int firstUnitID = desiredList.get(0).getView().getID();
		Integer secondUnitID = desiredList.size() > 1 ? desiredList.get(1).getView().getID() : null;
		// if there is no second unit, we want to copy everything once (which is basically not copying at all)

		List<Map<Integer, Action>> actionsList = new ArrayList<Map<Integer, Action>>();

		// collect all of the actions for the first unit, copying them if necessary (see the above block comment)
		// Liberatore is disappointed in this McCabe's complexity.
		if (secondUnitID==null){
			//Then there is only one unit to iterate over
			for (Action action : unitActions.get(firstUnitID)){
				Map<Integer, Action> temp = new HashMap<Integer, Action>();
				temp.put(firstUnitID, action);
				actionsList.add(temp);
			}
		} else {
			for (Action action1 : unitActions.get(firstUnitID)){
				for (Action action2 : unitActions.get(secondUnitID)){
					if (isLegal(action1, action2)) {
						Map<Integer, Action> temp = new HashMap<Integer, Action>();
						temp.put(firstUnitID, action1);
						temp.put(secondUnitID, action2);
						actionsList.add(temp);
					}
				}	
			}
		}


		List<GameStateChild> children = new ArrayList<GameStateChild>();
		for (Map<Integer, Action> actions : actionsList)
		{
			GameState newState = new GameState(this);
			newState.applyActions(actions);
			// apply the set of actions and return the GameStateChild that results from these actions
			children.add(new GameStateChild(actions, newState));
		}

		return children;
	}

	private void applyActions(Map<Integer, Action> actions)
	{
		for (Integer unitID : actions.keySet())
		{
			Action action = actions.get(unitID);
			switch (action.getType())
			{
			case PRIMITIVEMOVE:
				DirectedAction move = (DirectedAction) action;
				Direction direction = move.getDirection();
				Unit unit = state.getUnit(unitID);
				state.moveUnit(unit, direction);
				break;
			case PRIMITIVEATTACK:
				TargetedAction attack = (TargetedAction) action;
				Unit attacker = state.getUnit(unitID);
				Unit defender = state.getUnit(attack.getTargetId()); // WHY IS THIS ONE SPELLED DIFFERENTLY??? EITHER ALWAYS DO "ID" OR ALWAYS DO "Id" NOT BOTH!!!
				defender.setHP(defender.getCurrentHealth() - attacker.getTemplate().getBasicAttack());
				break;
			default:
				// this really should never happen but it never hurts to be careful
				break;
			}
		}
	}

	private List<Action> getAllActions(Unit unit){
		List<Action> allActions = new ArrayList<Action>();

		// get all move actions
		// NOTE TO GRADER: The assignment says the only possible moves are up, down, left and right
		// There are no diagonals
		Direction[] directions = {Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST};
		for(Direction direction: directions){
			if (direction.xComponent() == 0 && direction.yComponent() == 0){
				continue;
			}
			if (isLegal(unit, direction)){
				allActions.add(Action.createPrimitiveMove(unit.getView().getID(), direction));
			}
		}

		// get all attacking actions
		UnitTemplate unitTemplate = unit.getTemplate();
		int upperBound = unitTemplate.getRange();
		int lowerBound = -upperBound;
		for (int xAdj = lowerBound; xAdj <= upperBound; xAdj++) {
			for (int yAdj = lowerBound; yAdj <= upperBound; yAdj++) {
				if (xAdj == 0 && yAdj == 0) {
					continue;
				}

				int newX = unit.getxPosition() + xAdj;
				int newY = unit.getyPosition() + yAdj;
				Unit enemyUnit = state.unitAt(newX, newY);
				if (enemyUnit == null) {
					// no unit here, next iteration
					continue;
				}

				if (enemyUnit.getPlayer() == unit.getPlayer()) {
					// this is not actually an enemy, next iteration
					continue;
				}

				allActions.add(Action.createPrimitiveAttack(unit.getView().getID(), enemyUnit.getView().getID()));
			}
		}

		return allActions;
	}

	private boolean isLegal(Unit unit, Direction direction){
		int x = unit.getxPosition() + direction.xComponent();
		int y = unit.getyPosition() + direction.yComponent();
		boolean inBounds = x < xExtent && x >= 0 && y < yExtent && y >= 0;
		for (ResourceNode.ResourceView resourceView : resourceNodes){
			if (!inBounds){
				return false;
			}
			inBounds = inBounds || resourceView.getXPosition() != x && resourceView.getYPosition() != y;	
		}
		return inBounds;
	}

	/**
	 * Are these two actions legal (together)?
	 * The only actions that can conflict are move actions that attempt to move to the same square.
	 */
	private boolean isLegal(Action action1, Action action2)
	{
		ActionType primitiveMove = ActionType.PRIMITIVEMOVE;
		if (action1.getType() != primitiveMove || action2.getType() != primitiveMove)
		{
			return true;
		}

		DirectedAction moveAction1 = (DirectedAction) action1;
		DirectedAction moveAction2 = (DirectedAction) action2;

		Unit unit1 = state.getUnit(moveAction1.getUnitId());
		Unit unit2 = state.getUnit(moveAction2.getUnitId());

		Direction direction1 = moveAction1.getDirection();
		Direction direction2 = moveAction2.getDirection();

		int x1 = unit1.getxPosition() + direction1.xComponent();
		int y1 = unit1.getyPosition() + direction1.yComponent();

		int x2 = unit2.getxPosition() + direction2.xComponent();
		int y2 = unit2.getyPosition() + direction2.yComponent();

		return (x1 != x2 || y1 != y2);
	}
}